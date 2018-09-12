"""
This file contains each of the models used for this competition. Each model is
fully contained within its own class. To use a model, the class is instantiated
and then fitted or loaded using the corresponding method. Finally, predictions
are made using the `predict` method.

Author: Sam Waterbury
GitHub: https://github.com/samwaterbury/salt-identification
"""

import numpy as np
from sklearn.model_selection import train_test_split

from keras.models import Model, load_model
from keras.layers import Input, Add, Activation, BatchNormalization, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import adam, sgd

from utilities import get_iou_round1, get_iou_round2, lovasz_loss, get_optimal_cutoff

# ------------------------ (1) U-Net with ResNet blocks ---------------------- #


class UNetResNet:
    def __init__(self, parameters):
        self.params = parameters

        # Extract the parameters for this model
        self.final = self.params['final']
        self.use_saved_model = self.params['use_saved_weights']
        self.save_path = self.params['save_path']
        self.verbose = self.params['verbosity']
        self.k_folds = self.params['k_folds']
        self.ki = self.params['kernel_initializer']
        self.neurons_init = self.params['neurons_initial']
        self.batchnorm_momentum = self.params['batchnorm_momentum']

        self.optim_stage1 = self.params['optimizer_stage1']
        self.lr_stage1 = self.params['lr_stage1']
        self.optim_momentum_stage1 = self.params['optimizer_momentum_stage1']
        self.optim_stage2 = self.params['optimizer_stage2']
        self.lr_stage2 = self.params['lr_stage2']
        self.optim_momentum_stage2 = self.params['optimizer_momentum_stage2']

        self.early_stop_stage1 = self.params['early_stopping_stage1']
        self.early_stop_stage2 = self.params['early_stopping_stage2']
        self.batch_size_stage1 = self.params['batch_size_stage1']
        self.batch_size_stage2 = self.params['batch_size_stage2']
        self.epochs_stage1 = self.params['epochs_stage1']
        self.epochs_stage2 = self.params['epochs_stage2']
        self.dropout_expansion = self.params['dropout_expansion']
        self.dropout_ratios_expansion = self.params['dropout_ratios_expansion']
        self.dropout_contraction = self.params['dropout_contraction']
        self.dropout_ratios_contraction = self.params['dropout_ratios_contraction']

        # Construct the network used for the first round of training
        input_layer = Input(shape=(101, 101, 1))
        output_layer = self.build_model(input_layer)
        self.model = Model(input_layer, output_layer)

        self.optimal_cutoff = self.params['optimal_cutoff']

    def batch_activation(self, input_layer):
        layer = BatchNormalization(momentum=self.batchnorm_momentum)(input_layer)
        layer = Activation('relu')(layer)
        return layer

    def convolution_block(self, block_input, neurons, activation=True):
        conv = Conv2D(neurons, kernel_size=(3, 3), strides=(1, 1), padding='same',
                      kernel_initializer=self.ki)(block_input)
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
        # 101 -> 50
        conv1 = Conv2D(self.neurons_init, (3, 3), padding='same', kernel_initializer=self.ki)(input_layer)
        conv1 = self.residual_block(conv1, self.ki)
        conv1 = self.residual_block(conv1, self.neurons_init, batch_activation=True)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        if self.dropout_expansion:
            pool1 = Dropout(rate=self.dropout_ratios_expansion[0], seed=1)(pool1)

        # 50 -> 25
        conv2 = Conv2D(self.neurons_init * 2, (3, 3), padding='same', kernel_initializer=self.ki)(pool1)
        conv2 = self.residual_block(conv2, self.neurons_init * 2)
        conv2 = self.residual_block(conv2, self.neurons_init * 2, batch_activation=True)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        if self.dropout_expansion:
            pool2 = Dropout(rate=self.dropout_ratios_expansion[1])(pool2)

        # 25 -> 12
        conv3 = Conv2D(self.neurons_init * 4, (3, 3), padding='same', kernel_initializer=self.ki)(pool2)
        conv3 = self.residual_block(conv3, self.neurons_init * 4)
        conv3 = self.residual_block(conv3, self.neurons_init * 4, batch_activation=True)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        if self.dropout_expansion:
            pool3 = Dropout(rate=self.dropout_ratios_expansion[2])(pool3)

        # 12 -> 6
        conv4 = Conv2D(self.neurons_init * 8, (3, 3), padding='same', kernel_initializer=self.ki)(pool3)
        conv4 = self.residual_block(conv4, self.neurons_init * 8)
        conv4 = self.residual_block(conv4, self.neurons_init * 8, batch_activation=True)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
        if self.dropout_expansion:
            pool4 = Dropout(rate=self.dropout_ratios_expansion[3])(pool4)

        # Middle
        convm = Conv2D(self.neurons_init * 16, (3, 3), padding='same', kernel_initializer=self.ki)(pool4)
        convm = self.residual_block(convm, self.neurons_init * 16)
        convm = self.residual_block(convm, self.neurons_init * 16, batch_activation=True)

        # 6 -> 12
        deconv4 = Conv2DTranspose(self.neurons_init * 8, (3, 3), (2, 2), 'same', kernel_initializer=self.ki)(convm)
        uconv4 = concatenate([deconv4, conv4])
        if self.dropout_contraction:
            uconv4 = Dropout(rate=self.dropout_ratios_contraction[0])(uconv4)

        uconv4 = Conv2D(self.neurons_init * 8, (3, 3), padding='same', kernel_initializer=self.ki)(uconv4)
        uconv4 = self.residual_block(uconv4, self.neurons_init * 8)
        uconv4 = self.residual_block(uconv4, self.neurons_init * 8, batch_activation=True)

        # 12 -> 25
        deconv3 = Conv2DTranspose(self.neurons_init * 4, (3, 3), (2, 2), 'valid', kernel_initializer=self.ki)(uconv4)
        uconv3 = concatenate([deconv3, conv3])
        if self.dropout_contraction:
            uconv3 = Dropout(rate=self.dropout_ratios_contraction[1])(uconv3)

        uconv3 = Conv2D(self.neurons_init * 4, (3, 3), padding='same', kernel_initializer=self.ki)(uconv3)
        uconv3 = self.residual_block(uconv3, self.neurons_init * 4)
        uconv3 = self.residual_block(uconv3, self.neurons_init * 4, batch_activation=True)

        # 25 -> 50
        deconv2 = Conv2DTranspose(self.neurons_init * 2, (3, 3), (2, 2), 'same', kernel_initializer=self.ki)(uconv3)
        uconv2 = concatenate([deconv2, conv2])
        if self.dropout_contraction:
            uconv2 = Dropout(rate=self.dropout_ratios_contraction[2])(uconv2)

        uconv2 = Conv2D(self.neurons_init * 2, (3, 3), 'same', kernel_initializer=self.ki)(uconv2)
        uconv2 = self.residual_block(uconv2, self.neurons_init * 2)
        uconv2 = self.residual_block(uconv2, self.neurons_init * 2, batch_activation=True)

        # 50 -> 101
        deconv1 = Conv2DTranspose(self.neurons_init, (3, 3), (2, 2), 'valid', kernel_initializer=self.ki)(uconv2)
        uconv1 = concatenate([deconv1, conv1])
        if self.dropout_contraction:
            uconv1 = Dropout(rate=self.dropout_ratios_contraction[3])(uconv1)

        uconv1 = Conv2D(self.neurons_init, (3, 3), padding='same', kernel_initializer=self.ki)(uconv1)
        uconv1 = self.residual_block(uconv1, self.neurons_init)
        uconv1 = self.residual_block(uconv1, self.neurons_init, batch_activation=True)

        # Output layer
        output_layer = Conv2D(1, (1, 1), padding='same', kernel_initializer=self.ki)(uconv1)
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
        if self.use_saved_model:
            self.model = load_model(self.save_path)
            return

        if self.final:
            x_train = np.array(train['image'].tolist()).reshape(-1, 101, 101, 1)
            y_train = np.array(train['mask'].tolist()).reshape(-1, 101, 101, 1)
            valid_data = x_valid = y_valid = None
        else:
            x_train, x_valid, y_train, y_valid \
                = train_test_split(np.array(train['image'].tolist()).reshape(-1, 101, 101, 1),
                                   np.array(train['mask'].tolist()).reshape(-1, 101, 101, 1),
                                   test_size=1 / self.k_folds,
                                   stratify=train['coverage_class'],
                                   random_state=1)
            valid_data = [x_valid, y_valid]

        # Data augmentation
        x_train = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
        y_train = np.append(y_train, [np.fliplr(x) for x in y_train], axis=0)

        # Compile the model (stage 1)
        if self.optim_stage1 == 'adam':
            optim1 = adam(lr=self.lr_stage1)
        elif self.optim_stage1 == 'sgd':
            optim1 = sgd(lr=self.lr_stage1, momentum=self.optim_momentum_stage1)
        else:
            raise NotImplementedError('Optimizer should be either adam or SGD.')
        self.model.compile(optimizer=optim1, loss='binary_crossentropy', metrics=[get_iou_round1])

        # Callbacks
        callbacks = [
            EarlyStopping(monitor='get_iou_round1', patience=self.early_stop_stage1, verbose=self.verbose, mode='max'),
            ReduceLROnPlateau(monitor='get_iou_round1', factor=0.5, patience=5, verbose=self.verbose, mode='max',
                              min_lr=0.0001),
            ModelCheckpoint(self.save_path, monitor='get_iou_round1', mode='max', verbose=self.verbose,
                            save_best_only=True)
        ]

        # First stage of model fitting
        self.model.fit(x=x_train, y=y_train, batch_size=self.batch_size_stage1, epochs=self.epochs_stage1,
                       verbose=self.verbose, callbacks=callbacks, validation_data=valid_data)

        # Now drop the final activation and switch to Lovasz loss function
        stage2_model = Model(self.model.layers[0].input, self.model.layers[-1].input)

        # Compile the model (stage 2)
        if self.optim_stage2 == 'adam':
            optim2 = adam(lr=self.lr_stage1)
        elif self.optim_stage2 == 'sgd':
            optim2 = sgd(lr=self.lr_stage2, momentum=self.optim_momentum_stage2)
        else:
            raise NotImplementedError('Optimizer should be either adam or SGD.')
        stage2_model.compile(optimizer=optim2, loss=lovasz_loss, metrics=[get_iou_round2])

        # Update the callbacks
        for cb in callbacks:
            cb.monitor = 'get_iou_round2'

        stage2_model.fit(x=x_train, y=y_train, batch_size=self.batch_size_stage2, epochs=self.epochs_stage2,
                         verbose=self.verbose, callbacks=callbacks, validation_data=valid_data)
        self.model = stage2_model

        # Determine the optimal likelihood cutoff for segmenting images
        if not self.final:
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
        predictions = np.round(predictions > self.optimal_cutoff)

        return predictions / 2


# --------------------------------- (1) U-Net -------------------------------- #


# class UNet:
#     def __init__(self, save_path=None):
#         """
#         Create the model as soon as an instance of this class is made.
#         """
#         # Construct the network
#         self.model = self.build_model(neurons_init=16, kernel_init='he_normal')
#
#         # Specify the optimization scheme
#         self.model.compile(optimizer=adam(lr=0.01),
#                            loss='binary_crossentropy',
#                            metrics=['accuracy', competition_metric])
#
#         # Callbacks
#         self.callbacks = []
#         self.callbacks.append(
#             EarlyStopping(monitor='competition_metric', patience=10, verbose=1, mode='max')
#         )
#         self.callbacks.append(
#             ReduceLROnPlateau(monitor='competition_metric', factor=0.5, patience=5, verbose=1,
#                               mode='max', min_lr=0.0001)
#         )
#         if save_path is not None:
#             self.callbacks.append(
#                 ModelCheckpoint(save_path, monitor='competition_metric', mode='max', verbose=1,
#                                 save_best_only=True, save_weights_only=False)
#             )
#
#         # Parameters for fitting
#         self.batch_size = 32
#         self.epochs = 100
#         self.save_path = save_path
#
#     @staticmethod
#     def conv_contraction_block(block_input, neurons, kernal_init):
#         """
#         (Convolution -> Batch Normalization -> RELU) * 2 -> Max Pooling
#         """
#         # First round
#         conv1 = Conv2D(neurons, kernel_size=(3, 3), padding='same', kernel_initializer=kernal_init)(block_input)
#         conv1 = BatchNormalization()(conv1)
#         conv1 = Activation('relu')(conv1)
#
#         # Second round
#         conv2 = Conv2D(neurons, kernel_size=(3, 3), padding='same', kernel_initializer=kernal_init)(conv1)
#         conv2 = BatchNormalization()(conv2)
#         conv2 = Activation('relu')(conv2)
#
#         # Pooling function
#         pool = MaxPooling2D(pool_size=(2, 2), padding='valid')(conv2)
#         return pool, conv2
#
#     @staticmethod
#     def conv_static_block(block_input, neurons, kernal_init):
#         """
#         (Convolution -> Batch Normalization -> RELU) * 2
#         """
#         # First round
#         conv1 = Conv2D(neurons, kernel_size=(3, 3), padding='same', kernel_initializer=kernal_init)(block_input)
#         conv1 = BatchNormalization()(conv1)
#         conv1 = Activation('relu')(conv1)
#
#         # Second round
#         conv2 = Conv2D(neurons, kernel_size=(3, 3), padding='same', kernel_initializer=kernal_init)(conv1)
#         conv2 = BatchNormalization()(conv2)
#         conv2 = Activation('relu')(conv2)
#
#         # No pooling function for this block
#         return conv2
#
#     @staticmethod
#     def conv_expansion_block(block_input, corr_conv, neurons, kernel_init):
#         """
#         Upconvolution -> Concatenate -> (Convolution -> BN -> RELU) * 2
#         """
#         # Upsample & convolution
#         upconv = Conv2DTranspose(neurons, kernel_size=(3, 3), strides=(2, 2), padding='same')(block_input)
#
#         # Concatenate with corresponding convolution from expansion
#         conv = concatenate([upconv, corr_conv])
#
#         # First round
#         conv = Conv2D(neurons, kernel_size=(3, 3), padding='same', kernel_initializer=kernel_init)(conv)
#         conv = BatchNormalization()(conv)
#         conv = Activation('relu')(conv)
#
#         # Second round
#         conv = Conv2D(neurons, kernel_size=(3, 3), padding='same', kernel_initializer=kernel_init)(conv)
#         conv = BatchNormalization()(conv)
#         conv = Activation('relu')(conv)
#
#         # No pooling function for this block
#         return conv
#
#     @staticmethod
#     def build_model(neurons_init, kernel_init='he_normal'):
#         """
#         Input -> Contraction * 4 -> Middle -> Expansion * 4 -> Output
#         """
#         input_layer = Input(shape=(128, 128, 1), batch_shape=None)
#
#         # Image size 128x128 -> 8x8
#         conv1, corr_conv1 = UNet.conv_contraction_block(input_layer, neurons_init, kernel_init)
#         conv2, corr_conv2 = UNet.conv_contraction_block(conv1, neurons_init * 2, kernel_init)
#         conv3, corr_conv3 = UNet.conv_contraction_block(conv2, neurons_init * 4, kernel_init)
#         conv4, corr_conv4 = UNet.conv_contraction_block(conv3, neurons_init * 8, kernel_init)
#
#         # Middle block (size does not change here)
#         conv_middle = UNet.conv_static_block(conv4, neurons_init * 16, kernel_init)
#
#         # Image size 8x8 -> 128x128
#         convexp4 = UNet.conv_expansion_block(conv_middle, corr_conv4, neurons_init * 8, kernel_init)
#         convexp3 = UNet.conv_expansion_block(convexp4, corr_conv3, neurons_init * 4, kernel_init)
#         convexp2 = UNet.conv_expansion_block(convexp3, corr_conv2, neurons_init * 2, kernel_init)
#         convexp1 = UNet.conv_expansion_block(convexp2, corr_conv1, neurons_init, kernel_init)
#
#         # Output layer of 128x128x1 image tensors
#         output_layer = Conv2D(1, kernel_size=(1, 1), padding='same', activation='sigmoid')(convexp1)
#
#         return Model(input_layer, output_layer)
#
#     def fit_model(self, x_train, y_train, x_valid, y_valid):
#         """
#         Fit this instance's model using parameters defined in __init__.
#         """
#         self.model.fit(x=x_train, y=y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=1,
#                        callbacks=self.callbacks, validation_data=[x_valid, y_valid])
#
#     def load_model(self, load_path):
#         """
#         Load previously saved model.
#         """
#         self.model.load_weights(load_path)
#
#     def predict(self, x_test):
#         """
#         Wrapper for predictions made by this instance's model.
#         """
#         return self.model.predict(x_test, verbose=1)
