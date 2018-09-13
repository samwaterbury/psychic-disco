"""
This file contains each of the models used for this competition. Each model is
fully contained within its own class. To use a model, the class is instantiated
and then fitted or loaded using the corresponding method. Finally, predictions
are made using the `predict` method.

Author: Sam Waterbury
GitHub: https://github.com/samwaterbury/salt-identification
"""

import os

import numpy as np
from sklearn.model_selection import train_test_split

from keras.models import Model, load_model
from keras.layers import Input, Add, Activation, BatchNormalization, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import adam, sgd
from keras.utils.generic_utils import get_custom_objects

from utilities import iou_stage1, iou_stage2, lovasz_loss, get_optimal_cutoff

# ------------------------ (1) U-Net with ResNet blocks ---------------------- #


class UNetResNet(object):
    def __init__(self, parameters):
        """
        Load or construct the model using the parameters specified for this run.

        :param parameters: <parameters>['model_parameters']['unet_resnet'] dict.
        """
        self.params = parameters
        self.optimal_cutoff = None

        # Load the model if we can; then we are done in this method
        if self.params['use_saved_model'] and os.path.exists(self.params['save_path']):
            print('Loading saved U-Net with ResNet blocks...')
            get_custom_objects().update({'iou_stage1': iou_stage1})
            self.model = load_model(self.params['save_path'])
            self.optimal_cutoff = self.params['optimal_cutoff']

        # Otherwise, construct the model
        else:
            input_layer = Input(shape=(101, 101, 1))
            output_layer = self.build_model(input_layer)
            self.model = Model(input_layer, output_layer)

        # Now, if the model was loaded, we can make predictions.
        # Otherwise, we need to train the model before we can predict.

    def batch_activation(self, input_layer):
        layer = BatchNormalization(momentum=self.params['batchnorm_momentum'])(input_layer)
        layer = Activation('relu')(layer)
        return layer

    def convolution_block(self, block_input, neurons, activation=True):
        conv = Conv2D(neurons, kernel_size=(3, 3), strides=(1, 1), padding='same',
                      kernel_initializer=self.params['kernel_initializer'])(block_input)
        if activation:
            conv = self.batch_activation(conv)
        return conv

    def residual_block(self, block_input, neurons, batch_activation=False):
        layer = self.batch_activation(block_input)
        layer = self.convolution_block(layer, neurons)
        layer = self.convolution_block(layer, neurons, activation=False)
        layer = Add()([layer, block_input])
        if batch_activation:
            layer = self.batch_activation(layer)
        return layer

    def build_model(self, input_layer):
        """
        Constructs the U-Net arhitecture of layers using residual blocks.

        :param input_layer: Input(shape=(101, 101, 1)) object.
        :return: Output layer as a function of the entire network.
        """
        n_init = self.params['neurons_initial']
        ki = self.params['kernel_initializer']
        use_expansion_dropout = self.params['dropout_expansion']
        use_contraction_dropout = self.params['dropout_contraction']
        do_ratios_exp = self.params['dropout_ratios_expansion']
        do_ratios_con = self.params['dropout_ratios_contraction']

        # 101 -> 50
        c1 = Conv2D(n_init, (3, 3), padding='same', kernel_initializer=ki)(input_layer)
        c1 = self.residual_block(c1, n_init, ki)
        c1 = self.residual_block(c1, n_init, batch_activation=True)
        p1 = MaxPooling2D(pool_size=(2, 2))(c1)
        if use_expansion_dropout:
            p1 = Dropout(rate=do_ratios_exp[0], seed=1)(p1)

        # 50 -> 25
        c2 = Conv2D(n_init * 2, (3, 3), padding='same', kernel_initializer=ki)(p1)
        c2 = self.residual_block(c2, n_init * 2)
        c2 = self.residual_block(c2, n_init * 2, batch_activation=True)
        p2 = MaxPooling2D(pool_size=(2, 2))(c2)
        if use_expansion_dropout:
            p2 = Dropout(rate=do_ratios_exp[1])(p2)

        # 25 -> 12
        c3 = Conv2D(n_init * 4, (3, 3), padding='same', kernel_initializer=ki)(p2)
        c3 = self.residual_block(c3, n_init * 4)
        c3 = self.residual_block(c3, n_init * 4, batch_activation=True)
        p3 = MaxPooling2D(pool_size=(2, 2))(c3)
        if use_expansion_dropout:
            p3 = Dropout(rate=do_ratios_exp[2])(p3)

        # 12 -> 6
        c4 = Conv2D(n_init * 8, (3, 3), padding='same', kernel_initializer=ki)(p3)
        c4 = self.residual_block(c4, n_init * 8)
        c4 = self.residual_block(c4, n_init * 8, batch_activation=True)
        p4 = MaxPooling2D(pool_size=(2, 2))(c4)
        if use_expansion_dropout:
            p4 = Dropout(rate=do_ratios_exp[3])(p4)

        # Middle
        cm = Conv2D(n_init * 16, (3, 3), padding='same', kernel_initializer=ki)(p4)
        cm = self.residual_block(cm, n_init * 16)
        cm = self.residual_block(cm, n_init * 16, batch_activation=True)

        # 6 -> 12
        d4 = Conv2DTranspose(n_init * 8, (3, 3), strides=(2, 2), padding='same', kernel_initializer=ki)(cm)
        u4 = concatenate([d4, c4])
        if use_contraction_dropout:
            u4 = Dropout(rate=do_ratios_con[0])(u4)

        u4 = Conv2D(n_init * 8, (3, 3), padding='same', kernel_initializer=ki)(u4)
        u4 = self.residual_block(u4, n_init * 8)
        u4 = self.residual_block(u4, n_init * 8, batch_activation=True)

        # 12 -> 25
        d3 = Conv2DTranspose(n_init * 4, (3, 3), strides=(2, 2), padding='valid', kernel_initializer=ki)(u4)
        u3 = concatenate([d3, c3])
        if use_contraction_dropout:
            u3 = Dropout(rate=do_ratios_con[1])(u3)

        u3 = Conv2D(n_init * 4, (3, 3), padding='same', kernel_initializer=ki)(u3)
        u3 = self.residual_block(u3, n_init * 4)
        u3 = self.residual_block(u3, n_init * 4, batch_activation=True)

        # 25 -> 50
        d2 = Conv2DTranspose(n_init * 2, (3, 3), strides=(2, 2), padding='same', kernel_initializer=ki)(u3)
        u2 = concatenate([d2, c2])
        if use_contraction_dropout:
            u2 = Dropout(rate=do_ratios_con[2])(u2)

        u2 = Conv2D(n_init * 2, (3, 3), padding='same', kernel_initializer=ki)(u2)
        u2 = self.residual_block(u2, n_init * 2)
        u2 = self.residual_block(u2, n_init * 2, batch_activation=True)

        # 50 -> 101
        d1 = Conv2DTranspose(n_init, (3, 3), strides=(2, 2), padding='valid', kernel_initializer=ki)(u2)
        u1 = concatenate([d1, c1])
        if use_contraction_dropout:
            u1 = Dropout(rate=do_ratios_con[3])(u1)

        u1 = Conv2D(n_init, (3, 3), padding='same', kernel_initializer=ki)(u1)
        u1 = self.residual_block(u1, n_init)
        u1 = self.residual_block(u1, n_init, batch_activation=True)

        # Output layer
        output_layer = Conv2D(1, (1, 1), padding='same', kernel_initializer=ki)(u1)
        output_layer = Activation('sigmoid')(output_layer)

        return output_layer

    def fit_model(self, train):
        """
        Fit this model in two training stages. During the first stage, use
        binary crossentropy as the loss function. Then, during the second stage,
        drop the final activation on the output layer and continue fitting with
        the Lovasz loss function.

        Alternatively, load the saved model if the parameters specify to do so.

        :param train: Training DataFrame generated by `construct_data()`.
        """
        vb = self.params['verbose']

        # If these are final predictions, train on ALL the data
        if self.params['final']:
            x_train = np.array(train['image'].tolist()).reshape(-1, 101, 101, 1)
            y_train = np.array(train['mask'].tolist()).reshape(-1, 101, 101, 1)
            valid_data = x_valid = y_valid = None

        # Otherwise, partition the training data in to training/validation sets
        else:
            x_train, x_valid, y_train, y_valid \
                = train_test_split(np.array(train['image'].tolist()).reshape(-1, 101, 101, 1),
                                   np.array(train['mask'].tolist()).reshape(-1, 101, 101, 1),
                                   test_size=1 / self.params['k_folds'],
                                   stratify=train['coverage_class'],
                                   random_state=1)
            valid_data = [x_valid, y_valid]

        # Data augmentation (create mirrors around the vertical axis)
        x_train = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
        y_train = np.append(y_train, [np.fliplr(x) for x in y_train], axis=0)

        # TRAINING STAGE 1
        print('Fitting the Unet with ResNet blocks (stage 1)...')

        # Set the optimizer and compile the model
        if self.params['optimizer_stage1'] == 'adam':
            optim1 = adam(lr=self.params['lr_stage1'])
        elif self.params['optimizer_stage1'] == 'sgd':
            optim1 = sgd(lr=self.params['lr_stage1'], momentum=self.params['optimizer_momentum_stage1'])
        else:
            raise NotImplementedError('Optimizer should be either adam or SGD.')
        self.model.compile(optimizer=optim1, loss='binary_crossentropy', metrics=[iou_stage1])

        # Callbacks
        callbacks = [
            EarlyStopping(monitor='iou_stage1', patience=self.params['early_stopping'], verbose=vb, mode='max'),
            ReduceLROnPlateau(monitor='iou_stage1', factor=0.5, patience=5, verbose=vb, mode='max', min_lr=0.0001),
            ModelCheckpoint(self.params['save_path'], monitor='iou_stage1', mode='max', verbose=vb, save_best_only=True)
        ]

        # Fit the model
        self.model.fit(x=x_train, y=y_train, batch_size=self.params['batch_size_stage1'], shuffle=False,
                       epochs=self.params['epochs_stage1'], verbose=vb, callbacks=callbacks, validation_data=valid_data)

        # TRAINING STAGE 2
        print('Fitting the Unet with ResNet blocks (stage 2)...')

        # Now drop the final activation and switch to Lovasz loss function
        self.model = Model(self.model.layers[0].input, self.model.layers[-1].input)

        # Compile the model (stage 2)
        if self.params['optimizer_stage2'] == 'adam':
            optim2 = adam(lr=self.params['lr_stage2'])
        elif self.params['optimizer_stage2'] == 'sgd':
            optim2 = sgd(lr=self.params['lr_stage2'], momentum=self.params['optimizer_momentum_stage2'])
        else:
            raise NotImplementedError('Optimizer should be either adam or SGD.')
        self.model.compile(optimizer=optim2, loss=lovasz_loss, metrics=[iou_stage2])

        # Update the callbacks
        for cb in callbacks:
            cb.monitor = 'iou_stage2'

        self.model.fit(x=x_train, y=y_train, batch_size=self.params['batch_size_stage2'], shuffle=False,
                       epochs=self.params['epochs_stage2'], verbose=vb, callbacks=callbacks, validation_data=valid_data)

        # Determine the optimal likelihood cutoff for segmenting images
        if not self.params['final']:
            valid_predictions = self.model.predict(x_valid)
            self.optimal_cutoff = get_optimal_cutoff(valid_predictions, y_valid)

    def predict(self, x):
        """
        Make score predictions on `x`.

        :param x: Set of 101x101 image arrays.
        :return: 101x101 arrays of pixel scores for each image in `x`.
        """
        x_test_mirrored = np.array([np.fliplr(image) for image in x])
        predictions = self.model.predict(x).reshape(-1, 101, 101)
        predictions_mirrored = self.model.predict(x_test_mirrored).reshape(-1, 101, 101)
        predictions += np.array([np.fliplr(image) for image in predictions_mirrored])
        predictions = predictions / 2

        return predictions, self.optimal_cutoff


class ResNet32(object):
    def __init__(self):
        pass
