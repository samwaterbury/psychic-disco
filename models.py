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
from keras.layers import Input, Add, Activation, BatchNormalization, Dropout, SpatialDropout2D
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import adam, sgd
from keras.utils.generic_utils import get_custom_objects

from utilities import upsample, downsample, iou, iou_no_sigmoid, lovasz_loss, get_optimal_cutoff

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
            get_custom_objects().update({'iou': iou})
            self.model = load_model(self.params['save_path'])
            self.optimal_cutoff = self.params['optimal_cutoff']

        # Otherwise, construct the model
        else:
            input_layer = Input(shape=(101, 101, 1))
            output_layer = self.build_model(input_layer)
            self.model = Model(input_layer, output_layer)

        # Now, if the model was loaded, we can make predictions.
        # Otherwise, we need to train the model before we can predict.

    @staticmethod
    def batch_activation(input_layer, momentum):
        layer = BatchNormalization(momentum=momentum)(input_layer)
        layer = Activation('relu')(layer)
        return layer

    def convolution_block(self, block_input, neurons, activation=True):
        conv = Conv2D(neurons, kernel_size=(3, 3), strides=(1, 1), padding='same',
                      kernel_initializer=self.params['kernel_initializer'])(block_input)
        if activation:
            conv = self.batch_activation(conv, self.params['batchnorm_momentum'])
        return conv

    def residual_block(self, block_input, neurons, activation=False):
        layer = self.batch_activation(block_input, self.params['batchnorm_momentum'])
        layer = self.convolution_block(layer, neurons)
        layer = self.convolution_block(layer, neurons, activation=False)
        layer = Add()([layer, block_input])
        if activation:
            layer = self.batch_activation(layer, self.params['batchnorm_momentum'])
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
        c1 = self.residual_block(c1, n_init, activation=True)
        p1 = MaxPooling2D(pool_size=(2, 2))(c1)
        if use_expansion_dropout:
            p1 = Dropout(rate=do_ratios_exp[0], seed=1)(p1)

        # 50 -> 25
        c2 = Conv2D(n_init * 2, (3, 3), padding='same', kernel_initializer=ki)(p1)
        c2 = self.residual_block(c2, n_init * 2)
        c2 = self.residual_block(c2, n_init * 2, activation=True)
        p2 = MaxPooling2D(pool_size=(2, 2))(c2)
        if use_expansion_dropout:
            p2 = Dropout(rate=do_ratios_exp[1])(p2)

        # 25 -> 12
        c3 = Conv2D(n_init * 4, (3, 3), padding='same', kernel_initializer=ki)(p2)
        c3 = self.residual_block(c3, n_init * 4)
        c3 = self.residual_block(c3, n_init * 4, activation=True)
        p3 = MaxPooling2D(pool_size=(2, 2))(c3)
        if use_expansion_dropout:
            p3 = Dropout(rate=do_ratios_exp[2])(p3)

        # 12 -> 6
        c4 = Conv2D(n_init * 8, (3, 3), padding='same', kernel_initializer=ki)(p3)
        c4 = self.residual_block(c4, n_init * 8)
        c4 = self.residual_block(c4, n_init * 8, activation=True)
        p4 = MaxPooling2D(pool_size=(2, 2))(c4)
        if use_expansion_dropout:
            p4 = Dropout(rate=do_ratios_exp[3])(p4)

        # Middle
        cm = Conv2D(n_init * 16, (3, 3), padding='same', kernel_initializer=ki)(p4)
        cm = self.residual_block(cm, n_init * 16)
        cm = self.residual_block(cm, n_init * 16, activation=True)

        # 6 -> 12
        d4 = Conv2DTranspose(n_init * 8, (3, 3), strides=(2, 2), padding='same', kernel_initializer=ki)(cm)
        u4 = concatenate([d4, c4])
        if use_contraction_dropout:
            u4 = Dropout(rate=do_ratios_con[0])(u4)

        u4 = Conv2D(n_init * 8, (3, 3), padding='same', kernel_initializer=ki)(u4)
        u4 = self.residual_block(u4, n_init * 8)
        u4 = self.residual_block(u4, n_init * 8, activation=True)

        # 12 -> 25
        d3 = Conv2DTranspose(n_init * 4, (3, 3), strides=(2, 2), padding='valid', kernel_initializer=ki)(u4)
        u3 = concatenate([d3, c3])
        if use_contraction_dropout:
            u3 = Dropout(rate=do_ratios_con[1])(u3)

        u3 = Conv2D(n_init * 4, (3, 3), padding='same', kernel_initializer=ki)(u3)
        u3 = self.residual_block(u3, n_init * 4)
        u3 = self.residual_block(u3, n_init * 4, activation=True)

        # 25 -> 50
        d2 = Conv2DTranspose(n_init * 2, (3, 3), strides=(2, 2), padding='same', kernel_initializer=ki)(u3)
        u2 = concatenate([d2, c2])
        if use_contraction_dropout:
            u2 = Dropout(rate=do_ratios_con[2])(u2)

        u2 = Conv2D(n_init * 2, (3, 3), padding='same', kernel_initializer=ki)(u2)
        u2 = self.residual_block(u2, n_init * 2)
        u2 = self.residual_block(u2, n_init * 2, activation=True)

        # 50 -> 101
        d1 = Conv2DTranspose(n_init, (3, 3), strides=(2, 2), padding='valid', kernel_initializer=ki)(u2)
        u1 = concatenate([d1, c1])
        if use_contraction_dropout:
            u1 = Dropout(rate=do_ratios_con[3])(u1)

        u1 = Conv2D(n_init, (3, 3), padding='same', kernel_initializer=ki)(u1)
        u1 = self.residual_block(u1, n_init)
        u1 = self.residual_block(u1, n_init, activation=True)

        # Output layer
        output_layer = Conv2D(1, (1, 1), padding='same', kernel_initializer=ki)(u1)
        output_layer = Activation('sigmoid')(output_layer)

        return output_layer

    def train(self, train):
        """
        Fit this model in two training stages. During the first stage, use
        binary crossentropy as the loss function. Then, during the second stage,
        drop the final activation on the output layer and continue fitting with
        the Lovasz loss function.

        Alternatively, load the saved model if the parameters specify to do so.

        :param train: Training DataFrame generated by `construct_data()`.
        """
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
        self.model.compile(optimizer=optim1, loss='binary_crossentropy', metrics=[iou])

        # Callbacks
        callbacks = [
            EarlyStopping(monitor='iou', patience=self.params['early_stopping'], verbose=2, mode='max'),
            ReduceLROnPlateau(monitor='iou', factor=0.5, patience=5, verbose=2, mode='max', min_lr=0.0001),
            ModelCheckpoint(self.params['save_path'], monitor='iou', mode='max', verbose=2, save_best_only=True)
        ]

        # Fit the model
        self.model.fit(x=x_train, y=y_train, batch_size=self.params['batch_size_stage1'], shuffle=False,
                       epochs=self.params['epochs_stage1'], verbose=2, callbacks=callbacks, validation_data=valid_data)

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
        self.model.compile(optimizer=optim2, loss=lovasz_loss, metrics=[iou_no_sigmoid])

        # Update the callbacks
        for cb in callbacks:
            cb.monitor = 'iou_no_sigmoid'

        self.model.fit(x=x_train, y=y_train, batch_size=self.params['batch_size_stage2'], shuffle=False,
                       epochs=self.params['epochs_stage2'], verbose=2, callbacks=callbacks, validation_data=valid_data)

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
        x = np.array(x.tolist()).reshape(-1, 101, 101, 1)
        x_mirrored = np.array([np.fliplr(image) for image in x])
        predictions = self.model.predict(x).reshape(-1, 101, 101)
        predictions_mirrored = self.model.predict(x_mirrored).reshape(-1, 101, 101)
        predictions += np.array([np.fliplr(image) for image in predictions_mirrored])
        predictions = predictions / 2

        return predictions, self.optimal_cutoff


# -------------------------------- (2) ResNet50 ------------------------------ #


class ResNet50(object):
    def __init__(self, parameters):
        """
        Load or construct the model using the parameters specified for this run.

        :param parameters: <parameters>['model_parameters']['resnet50'] dict.
        """
        self.params = parameters
        self.optimal_cutoff = None

        # Load the model if we can; then we are done in this method
        if self.params['use_saved_model'] and os.path.exists(self.params['save_path']):
            print('Loading saved ResNet34 model...')
            self.model = load_model(self.params['save_path'])
            self.optimal_cutoff = self.params['optimal_cutoff']

        # Otherwise, construct the model
        else:
            self.base = ResNet50Base(self.params['pretrained_weights_path']).model
            for layer in self.base.layers:
                layer.trainable = True
            self.output_layer = self.build_network(self.base)
            self.model = Model(self.base.input, self.output_layer)

        # Now, if the model was loaded, we can make predictions.
        # Otherwise, we need to train the model before we can predict.

    @staticmethod
    def convolution_block(block_input, filters, prefix,
                          kernel_initializer='he_normal', use_batchnorm=True, batchnorm_momentum=0.99):
        """
        Convolution -> (Batch Normalization) -> RELU
        """
        layer = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', kernel_initializer=kernel_initializer,
                       name='{}_conv'.format(prefix))(block_input)
        if use_batchnorm:
            layer = BatchNormalization(momentum=batchnorm_momentum, name='{}_bn'.format(prefix))(layer)
        layer = Activation('relu', name='{}_activation'.format(prefix))(layer)
        return layer

    @staticmethod
    def build_network(base):
        c1 = base.get_layer("activation_1").output
        c2 = base.get_layer("activation_10").output
        c3 = base.get_layer("activation_22").output
        c4 = base.get_layer("activation_40").output
        c5 = base.get_layer("activation_49").output

        u6 = concatenate([UpSampling2D()(c5), c4])
        c6 = ResNet50.convolution_block(u6, filters=256, prefix='conv6_1')
        c6 = ResNet50.convolution_block(c6, filters=256, prefix='conv6_2')

        u7 = concatenate([UpSampling2D()(c6), c3])
        c7 = ResNet50.convolution_block(u7, filters=192, prefix='conv7_1')
        c7 = ResNet50.convolution_block(c7, filters=192, prefix='conv7_2')

        u8 = concatenate([UpSampling2D()(c7), c2])
        c8 = ResNet50.convolution_block(u8, filters=128, prefix='conv8_1')
        c8 = ResNet50.convolution_block(c8, filters=128, prefix='conv8_2')

        u9 = concatenate([UpSampling2D()(c8), c1])
        c9 = ResNet50.convolution_block(u9, filters=64, prefix='conv9_1')
        c9 = ResNet50.convolution_block(c9, filters=64, prefix='conv9_2')

        u10 = UpSampling2D()(c9)
        c10 = ResNet50.convolution_block(u10, filters=32, prefix='conv10_1')
        c10 = ResNet50.convolution_block(c10, filters=32, prefix='conv10_2')

        c10 = SpatialDropout2D(0.2)(c10)
        output_layer = Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(c10)

        return output_layer

    def train(self, train):
        # If these are final predictions, train on ALL the data
        if self.params['final']:
            x_train = np.array(train['image'].map(upsample).tolist()).reshape(-1, 101, 101, 3)
            y_train = np.array(train['mask'].map(upsample).tolist()).reshape(-1, 101, 101, 3)
            valid_data = x_valid = y_valid = None
            # Convert to 3 identical channels
            x_train = np.repeat(x_train, 3, axis=3)

        # Otherwise, partition the training data in to training/validation sets
        else:
            x_train, x_valid, y_train, y_valid \
                = train_test_split(np.array(train['image'].map(upsample).tolist()).reshape(-1, 101, 101, 3),
                                   np.array(train['mask'].map(upsample).tolist()).reshape(-1, 101, 101, 3),
                                   test_size=1 / self.params['k_folds'],
                                   stratify=train['coverage_class'],
                                   random_state=1)
            # Convert to 3 identical channels
            x_train = np.repeat(x_train, 3, axis=3)
            x_valid = np.repeat(x_valid, 3, axis=3) if x_valid is not None else None
            valid_data = [x_valid, y_valid]

        # Data augmentation (create mirrors around the vertical axis)
        x_train = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
        y_train = np.append(y_train, [np.fliplr(x) for x in y_train], axis=0)

        # TRAINING
        print('Fitting the ResNet34 model...')
        if self.params['optimizer'] == 'adam':
            optim = adam(lr=self.params['lr'])
        elif self.params['optimizer'] == 'sgd':
            optim = sgd(lr=self.params['lr'], momentum=self.params['optimizer_momentum'])
        else:
            raise NotImplementedError('Optimizer should be either adam or SGD.')
        self.model.compile(optimizer=optim, loss=lovasz_loss, metrics=[iou])

        # Callbacks
        callbacks = [
            EarlyStopping(monitor='iou', patience=self.params['early_stopping'], verbose=2, mode='max'),
            ReduceLROnPlateau(monitor='iou', factor=0.1, patience=4, verbose=2, mode='max', min_lr=0.00001),
            ModelCheckpoint(self.params['save_path'], monitor='iou', mode='max', verbose=2, save_best_only=True)
        ]

        # Fit the model
        self.model.fit(x=x_train, y=y_train, batch_size=self.params['batch_size'], shuffle=False,
                       epochs=self.params['epochs'], verbose=2, callbacks=callbacks, validation_data=valid_data)

        # Determine the optimal likelihood cutoff for segmenting images
        if not self.params['final']:
            x_valid_mirrored = np.array([np.fliplr(image) for image in x_valid])
            valid_predictions = self.model.predict(x_valid).reshape(-1, 128, 128)
            valid_predictions_mirrored = self.model.predict(x_valid_mirrored).reshape(-1, 128, 128)
            valid_predictions += np.array([np.fliplr(image) for image in valid_predictions_mirrored])
            valid_predictions = valid_predictions / 2
            valid_predictions = [downsample(y) for y in valid_predictions]

            self.optimal_cutoff = get_optimal_cutoff(valid_predictions, y_valid)

    def predict(self, x):
        """
        Make score predictions on `x`.

        :param x: Set of 101x101 image arrays.
        :return: 101x101 arrays of pixel scores for each image in `x`.
        """
        x = np.array(x.map(upsample).tolist()).reshape(-1, 128, 128, 1)
        x_mirrored = np.array([np.fliplr(image) for image in x])
        predictions = self.model.predict(x).reshape(-1, 128, 128)
        predictions_mirrored = self.model.predict(x_mirrored).reshape(-1, 128, 128)
        predictions += np.array([np.fliplr(image) for image in predictions_mirrored])
        predictions = predictions / 2
        predictions = [downsample(y) for y in predictions]

        return predictions, self.optimal_cutoff


class ResNet50Base(object):
    def __init__(self, weights_path):
        self.input_layer = Input(shape=(128, 128, 3))
        self.output_layer = ResNet50Base.build_network(self.input_layer)
        self.model = Model(self.input_layer, self.output_layer)
        self.model.load_weights(weights_path, by_name=True)

    @staticmethod
    def convolution_block(block_input, filters, kernel_size, stage, block, strides=(2, 2)):
        conv_name = 'res{}{}_branch'.format(str(stage), block) + '{}'
        batchnorm_name = 'bn{}{}_branch'.format(str(stage), block) + '{}'

        layer = Conv2D(filters[0], kernel_size=(1, 1), strides=strides, name=conv_name.format('2a'))(block_input)
        layer = BatchNormalization(axis=1, name=batchnorm_name.format('2a'))(layer)
        layer = Activation('relu')(layer)

        layer = Conv2D(filters[1], kernel_size=kernel_size, padding='same', name=conv_name.format('2b'))(layer)
        layer = BatchNormalization(axis=1, name=batchnorm_name.format('2b'))(layer)
        layer = Activation('relu')(layer)

        layer = Conv2D(filters[2], kernel_size=(1, 1), name=conv_name.format('2c'))(layer)
        layer = BatchNormalization(axis=1, name=batchnorm_name.format('2c'))(layer)

        direct_route = Conv2D(filters[2], kernel_size=(1, 1), strides=strides, name=conv_name.format('1'))(block_input)
        direct_route = BatchNormalization(axis=1, name=batchnorm_name.format('1'))(direct_route)

        layer = Add()([layer, direct_route])
        layer = Activation('relu')(layer)
        return layer

    @staticmethod
    def identity_block(block_input, filters, kernel_size, stage, block):
        conv_name = 'res{}{}_branch'.format(str(stage), block) + '{}'
        batchnorm_name = 'bn{}{}_branch'.format(str(stage), block) + '{}'

        layer = Conv2D(filters[0], kernel_size=(1, 1), name=conv_name.format('2a'))(block_input)
        layer = BatchNormalization(axis=1, name=batchnorm_name.format('2a'))(layer)
        layer = Activation('relu')(layer)

        layer = Conv2D(filters[1], kernel_size, padding='same', name=conv_name.format('2b'))(layer)
        layer = BatchNormalization(axis=1, name=batchnorm_name.format('2b'))(layer)
        layer = Activation('relu')(layer)

        layer = Conv2D(filters[2], (1, 1), name=conv_name.format('2c'))(layer)
        layer = BatchNormalization(axis=1, name=batchnorm_name.format('2c'))(layer)

        layer = Add()([layer, block_input])
        layer = Activation('relu')(layer)
        return layer

    @staticmethod
    def build_network(input_layer):
        layer = Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), padding='same', name='conv1')(input_layer)
        layer = BatchNormalization(axis=1, name='bn_conv1')(layer)
        layer = Activation('relu')(layer)
        layer = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(layer)

        layer = ResNet50Base.convolution_block(layer, filters=[64, 64, 256], strides=(1, 1), kernel_size=3, stage=2,
                                               block='a')
        layer = ResNet50Base.identity_block(layer, filters=[64, 64, 256], kernel_size=3, stage=2, block='b')
        layer = ResNet50Base.identity_block(layer, filters=[64, 64, 256], kernel_size=3, stage=2, block='c')

        layer = ResNet50Base.convolution_block(layer, filters=[128, 128, 512], kernel_size=3, stage=3, block='a')
        layer = ResNet50Base.identity_block(layer, filters=[128, 128, 512], kernel_size=3, stage=3, block='b')
        layer = ResNet50Base.identity_block(layer, filters=[128, 128, 512], kernel_size=3, stage=3, block='c')
        layer = ResNet50Base.identity_block(layer, filters=[128, 128, 512], kernel_size=3, stage=3, block='d')

        layer = ResNet50Base.convolution_block(layer, filters=[256, 256, 1024], kernel_size=3, stage=4, block='a')
        layer = ResNet50Base.identity_block(layer, filters=[256, 256, 1024], kernel_size=3, stage=4, block='b')
        layer = ResNet50Base.identity_block(layer, filters=[256, 256, 1024], kernel_size=3, stage=4, block='c')
        layer = ResNet50Base.identity_block(layer, filters=[256, 256, 1024], kernel_size=3, stage=4, block='d')
        layer = ResNet50Base.identity_block(layer, filters=[256, 256, 1024], kernel_size=3, stage=4, block='e')
        layer = ResNet50Base.identity_block(layer, filters=[256, 256, 1024], kernel_size=3, stage=4, block='f')

        layer = ResNet50Base.convolution_block(layer, filters=[512, 512, 2048], kernel_size=3, stage=5, block='a')
        layer = ResNet50Base.identity_block(layer, filters=[512, 512, 2048], kernel_size=3, stage=5, block='b')
        layer = ResNet50Base.identity_block(layer, filters=[512, 512, 2048], kernel_size=3, stage=5, block='c')

        return layer
