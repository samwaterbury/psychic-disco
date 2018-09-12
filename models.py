"""
This file contains each of the models used for this competition. Each model is
fully contained within its own class. To use a model, the class is instantiated
and then fitted or loaded using the corresponding method. Finally, predictions
are made using the `predict` method.

Author: Sam Waterbury
GitHub: https://github.com/samwaterbury/salt-identification
"""

import numpy as np

from keras.models import Model
from keras.layers import Input, Add, Activation, BatchNormalization, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import adam

from utilities import get_iou_round1, get_iou_round2, lovasz_loss

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

# --------------------------------- (1) U-Net -------------------------------- #


class UNetResNet:
    def __init__(self, save_path=None, dropout_ratio=0.5):

        # Construct the network used for the first round of training
        input_layer = Input(shape=(101, 101, 1))
        output_layer = self.build_model(input_layer, neurons_init=16, kernel_init='glorot_uniform',
                                        dropout_ratio=dropout_ratio)
        self.model = Model(input_layer, output_layer)

        # Parameters for fitting
        self.batch_size = 32
        self.epochs = 50
        self.early_stopping_patience = 20
        self.save_path = save_path

    @staticmethod
    def batch_activation(input_layer):
        layer = BatchNormalization(momentum=0.99)(input_layer)
        layer = Activation('relu')(layer)
        return layer

    @staticmethod
    def convolution_block(block_input, neurons, size, strides=(1, 1), padding='same', activation=True):
        conv = Conv2D(neurons, kernel_size=size, strides=strides, padding=padding,
                      kernel_initializer='glorot_uniform')(block_input)
        if activation:
            conv = UNetResNet.batch_activation(conv)
        return conv

    @staticmethod
    def residual_block(block_input, neurons=16, batch_activation=False):
        layer = UNetResNet.batch_activation(block_input)
        layer = UNetResNet.convolution_block(layer, neurons, size=(3, 3))
        layer = UNetResNet.convolution_block(layer, neurons, size=(3, 3), activation=False)
        layer = Add()([layer, block_input])
        if batch_activation:
            layer = UNetResNet.batch_activation(layer)
        return layer

    @staticmethod
    def build_model(input_layer, neurons_init=16, kernel_init='glorot_uniform', dropout_ratio=0.5):

        # 101 -> 50
        conv1 = Conv2D(neurons_init, kernel_size=(3, 3), strides=(1, 1), padding='same',
                       kernel_initializer=kernel_init)(input_layer)
        conv1 = UNetResNet.residual_block(conv1, neurons_init)
        conv1 = UNetResNet.residual_block(conv1, neurons_init, batch_activation=True)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        pool1 = Dropout(rate=dropout_ratio / 2, seed=1)(pool1)  # Is this good?

        # 50 -> 25
        conv2 = Conv2D(neurons_init * 2, kernel_size=(3, 3), padding='same', kernel_initializer=kernel_init)(pool1)
        conv2 = UNetResNet.residual_block(conv2, neurons_init * 2)
        conv2 = UNetResNet.residual_block(conv2, neurons_init * 2, batch_activation=True)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        pool2 = Dropout(rate=dropout_ratio)(pool2)

        # 25 -> 12
        conv3 = Conv2D(neurons_init * 4, kernel_size=(3, 3), padding='same', kernel_initializer=kernel_init)(pool2)
        conv3 = UNetResNet.residual_block(conv3, neurons_init * 4)
        conv3 = UNetResNet.residual_block(conv3, neurons_init * 4, batch_activation=True)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        pool3 = Dropout(rate=dropout_ratio)(pool3)

        # 12 -> 6
        conv4 = Conv2D(neurons_init * 8, kernel_size=(3, 3), padding='same', kernel_initializer=kernel_init)(pool3)
        conv4 = UNetResNet.residual_block(conv4, neurons_init * 8)
        conv4 = UNetResNet.residual_block(conv4, neurons_init * 8, batch_activation=True)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
        pool4 = Dropout(rate=dropout_ratio)(pool4)

        # Middle
        convm = Conv2D(neurons_init * 16, kernel_size=(3, 3), padding='same', kernel_initializer=kernel_init)(pool4)
        convm = UNetResNet.residual_block(convm, neurons_init * 16)
        convm = UNetResNet.residual_block(convm, neurons_init * 16, batch_activation=True)

        # 6 -> 12
        deconv4 = Conv2DTranspose(neurons_init * 8, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                  kernel_initializer=kernel_init)(convm)
        uconv4 = concatenate([deconv4, conv4])
        uconv4 = Dropout(rate=dropout_ratio)(uconv4)

        uconv4 = Conv2D(neurons_init * 8, kernel_size=(3, 3), padding='same',
                        kernel_initializer=kernel_init)(uconv4)
        uconv4 = UNetResNet.residual_block(uconv4, neurons_init * 8)
        uconv4 = UNetResNet.residual_block(uconv4, neurons_init * 8, batch_activation=True)

        # 12 -> 25
        deconv3 = Conv2DTranspose(neurons_init * 4, kernel_size=(3, 3), strides=(2, 2), padding='valid',  # TODO 'same'?
                                  kernel_initializer=kernel_init)(uconv4)
        uconv3 = concatenate([deconv3, conv3])
        uconv3 = Dropout(rate=dropout_ratio)(uconv3)

        uconv3 = Conv2D(neurons_init * 4, kernel_size=(3, 3), padding='same',
                        kernel_initializer=kernel_init)(uconv3)
        uconv3 = UNetResNet.residual_block(uconv3, neurons_init * 4)
        uconv3 = UNetResNet.residual_block(uconv3, neurons_init * 4, batch_activation=True)

        # 25 -> 50
        deconv2 = Conv2DTranspose(neurons_init * 2, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                  kernel_initializer=kernel_init)(uconv3)
        uconv2 = concatenate([deconv2, conv2])
        uconv2 = Dropout(rate=dropout_ratio)(uconv2)

        uconv2 = Conv2D(neurons_init * 2, kernel_size=(3, 3), padding='same',
                        kernel_initializer=kernel_init)(uconv2)
        uconv2 = UNetResNet.residual_block(uconv2, neurons_init * 2)
        uconv2 = UNetResNet.residual_block(uconv2, neurons_init * 2, batch_activation=True)

        # 50 -> 101
        deconv1 = Conv2DTranspose(neurons_init, kernel_size=(3, 3), strides=(2, 2), padding='valid',
                                  kernel_initializer=kernel_init)(uconv2)
        uconv1 = concatenate([deconv1, conv1])
        uconv1 = Dropout(rate=dropout_ratio)(uconv1)

        uconv1 = Conv2D(neurons_init, kernel_size=(3, 3), padding='same',
                        kernel_initializer=kernel_init)(uconv1)
        uconv1 = UNetResNet.residual_block(uconv1, neurons_init)
        uconv1 = UNetResNet.residual_block(uconv1, neurons_init, batch_activation=True)

        # Output layer
        output_layer = Conv2D(filters=1, kernel_size=(1, 1), padding='same', kernel_initializer=kernel_init)(uconv1)
        output_layer = Activation('sigmoid')(output_layer)

        return output_layer

    def fit_model(self, x_train, y_train, x_valid, y_valid):
        """
        Fit this model in two training stages. During the first stage, use
        binary crossentropy as the loss function. Then, during the second stage,
        drop the final activation on the output layer and continue fitting with
        the Lovasz loss function.

        :param x_train: Training set of 101x101x1 image arrays.
        :param y_train: Training set of 101x101x1 mask arrays.
        :param x_valid: Validation set of 101x101x1 image arrays.
        :param y_valid: Validation set of 101x101x1 mask arrays.
        :return:
        """

        # Compile the model
        self.model.compile(optimizer=adam(lr=0.01),
                           loss='binary_crossentropy',
                           metrics=[get_iou_round1])

        # Callbacks
        callbacks = [
            EarlyStopping(monitor='get_iou_round1', patience=self.early_stopping_patience, verbose=1, mode='max'),
            ReduceLROnPlateau(monitor='get_iou_round1', factor=0.5, patience=5, verbose=1, mode='max', min_lr=0.0001)
        ]

        # Only save this model if it wasn't loaded from a file
        if self.save_path is not None:
            callbacks.append(ModelCheckpoint(self.save_path, monitor='get_iou_round1', mode='max', verbose=1,
                                             save_best_only=True, save_weights_only=False))

        # First stage of model fitting
        self.model.fit(x=x_train, y=y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=1,
                       callbacks=callbacks, validation_data=[x_valid, y_valid])

        # Now drop the final activation and switch to Lovasz loss function
        stage2_model = Model(self.model.layers[0].input, self.model.layers[-1].input)
        stage2_model.compile(optimizer=adam(lr=0.01),
                             loss=lovasz_loss,
                             metrics=[get_iou_round2])

        # Update the callbacks
        for cb in callbacks:
            cb.monitor = 'get_iou_round2'

        stage2_model.fit(x=x_train, y=y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=1,
                         callbacks=callbacks, validation_data=[x_valid, y_valid])
        self.model = stage2_model

    def load_model(self, load_path):
        """
        Loads the model weights from a saved weights file instead of retraining.

        :param load_path: Path to 'unetresnet.model' file.
        """
        self.model = Model(self.model.layers[0].input, self.model.layers[-1].input)
        self.model.compile(optimizer=adam(lr=0.01),
                           loss=lovasz_loss,
                           metrics=[get_iou_round2])
        self.model.load_weights(load_path)

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
        return predictions / 2
