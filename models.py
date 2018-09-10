"""
This file constructs the model upon a call to `get_fitted_model()`.

Author: Sam Waterbury
GitHub: https://github.com/samwaterbury/salt-identification
"""

from keras.optimizers import SGD, adam
from keras.models import Model, load_model
from keras.layers import Input, Dropout, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from utilities import competition_metric


# --------------------------------- (1) U-Net -------------------------------- #


class UNet:
    def __init__(self, save_path):
        """
        Create the model as soon as an instance of this class is made.
        """
        # Construct the network
        self.model = self.build_model(self, neurons_init=16, kernel_init='he_normal')

        # Specify the optimization scheme
        self.model.compile(optimizer=adam(lr=0.01), loss='binary_crossentropy',
                           metrics=['accuracy', competition_metric])

        # Callbacks
        es = EarlyStopping(monitor='competition_metric', patience=10, verbose=1, mode='max')
        mc = ModelCheckpoint(save_path, monitor='competition_metric', mode='max', verbose=1,
                             save_best_only=True, save_weights_only=False)
        lr = ReduceLROnPlateau(monitor='competition_metric', factor=0.5, patience=5, verbose=1,
                               mode='max', min_lr=0.0001)

        # Parameters for fitting
        self.batch_size = 32
        self.epochs = 50
        self.callbacks = [es, mc, lr]

    @staticmethod
    def conv_contraction_block(block_input, neurons, kernal_init):
        """
        (Convolution -> Batch Normalization -> RELU) * 2 -> Max Pooling
        """
        # First round
        conv1 = Conv2D(neurons, kernel_size=(3, 3), padding='same', kernel_initializer=kernal_init)(block_input)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)

        # Second round
        conv2 = Conv2D(neurons, kernel_size=(3, 3), padding='same', kernel_initializer=kernal_init)(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)

        # Pooling function
        pool = MaxPooling2D(pool_size=(2, 2), padding='valid')(conv2)
        return pool, conv2

    @staticmethod
    def conv_static_block(block_input, neurons, kernal_init):
        """
        (Convolution -> Batch Normalization -> RELU) * 2
        """
        # First round
        conv1 = Conv2D(neurons, kernel_size=(3, 3), padding='same', kernel_initializer=kernal_init)(block_input)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)

        # Second round
        conv2 = Conv2D(neurons, kernel_size=(3, 3), padding='same', kernel_initializer=kernal_init)(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)

        # No pooling function for this block
        return conv2

    @staticmethod
    def conv_expansion_block(block_input, corr_conv, neurons, kernel_init):
        """
        Upconvolution -> Concatenate -> (Convolution -> BN -> RELU) * 2
        """
        # Upsample & convolution
        upconv = Conv2DTranspose(neurons, kernel_size=(3, 3), strides=(2, 2), padding='same')(block_input)

        # Concatenate with corresponding convolution from expansion
        conv = concatenate([upconv, corr_conv])

        # First round
        conv = Conv2D(neurons, kernel_size=(3, 3), padding='same', kernel_initializer=kernel_init)(conv)
        conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)

        # Second round
        conv = Conv2D(neurons, kernel_size=(3, 3), padding='same', kernel_initializer=kernel_init)(conv)
        conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)

        # No pooling function for this block
        return conv

    @staticmethod
    def build_model(self, neurons_init, kernel_init='he_normal'):
        """
        Input -> Contraction * 4 -> Middle -> Expansion * 4 -> Output
        """
        input_layer = Input(shape=(128, 128, 1), batch_shape=None)

        # Image size 128x128 -> 8x8
        conv1, corr_conv1 = self.conv_contraction_block(neurons_init, input_layer, kernel_init)
        conv2, corr_conv2 = self.conv_contraction_block(neurons_init * 2, conv1, kernel_init)
        conv3, corr_conv3 = self.conv_contraction_block(neurons_init * 4, conv2, kernel_init)
        conv4, corr_conv4 = self.conv_contraction_block(neurons_init * 8, conv3, kernel_init)

        # Middle block (size does not change here)
        conv_middle = self.conv_static_block(neurons_init * 16, conv4, kernel_init)

        # Image size 8x8 -> 128x128
        convexp4 = self.conv_expansion_block(neurons_init * 8, conv_middle, corr_conv4, kernel_init)
        convexp3 = self.conv_expansion_block(neurons_init * 4, convexp4, corr_conv3, kernel_init)
        convexp2 = self.conv_expansion_block(neurons_init * 2, convexp3, corr_conv2, kernel_init)
        convexp1 = self.conv_expansion_block(neurons_init, convexp2, corr_conv1, kernel_init)

        # Output layer of 128x128x1 image tensors
        output_layer = Conv2D(1, kernel_size=(1, 1), padding='same', activation='sigmoid')(convexp1)

        return Model(input_layer, output_layer)

    def fit_model(self, x_train, y_train, x_valid, y_valid):
        """
        Fit this instance's model using parameters defined in __init__.
        """
        self.model.fit(x=x_train, y=y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=1,
                       callbacks=self.callbacks, validation_data=[x_valid, y_valid])

    def predict(self, x_test):
        """
        Wrapper for predictions made by this instance's model.
        """
        return self.model.predict(x_test)
