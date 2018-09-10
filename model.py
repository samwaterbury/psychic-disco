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


def convolution_block(neurons, block_input, kernel_initializer='he_normal'):
    conv1 = Conv2D(neurons, kernel_size=(3, 3), padding='same', kernel_initializer=kernel_initializer)(block_input)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv2 = Conv2D(neurons, kernel_size=(3, 3), padding='same', kernel_initializer=kernel_initializer)(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    pool = MaxPooling2D(pool_size=(2, 2), padding='valid')(conv2)
    return pool, conv2


def middle_convolution_block(neurons, block_input, kernel_initializer='he_normal'):
    conv1 = Conv2D(neurons, kernel_size=(3, 3), padding='same', kernel_initializer=kernel_initializer)(block_input)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv2 = Conv2D(neurons, kernel_size=(3, 3), padding='same', kernel_initializer=kernel_initializer)(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    return conv2


def upconvolution_block(neurons, block_input, corr_conv, kernel_initializer='he_normal'):
    deconv = Conv2DTranspose(neurons, kernel_size=(3, 3), strides=(2, 2), padding='same')(block_input)
    upconv = concatenate([deconv, corr_conv])
    upconv = Conv2D(neurons, kernel_size=(3, 3), padding='same', kernel_initializer=kernel_initializer)(upconv)
    upconv = BatchNormalization()(upconv)
    upconv = Activation('relu')(upconv)
    upconv = Conv2D(neurons, kernel_size=(3, 3), padding='same', kernel_initializer=kernel_initializer)(upconv)
    upconv = BatchNormalization()(upconv)
    upconv = Activation('relu')(upconv)
    return upconv


def build_network(input_layer, initial_neurons, kernel_initializer = 'he_normal'):
    """Builds the network and returns the final output layer."""

    # 128 -> 64
    conv1, corr_conv1 = convolution_block(initial_neurons, input_layer, kernel_initializer)

    # 64 -> 32
    conv2, corr_conv2 = convolution_block(initial_neurons * 2, conv1, kernel_initializer)

    # 32 -> 16
    conv3, corr_conv3 = convolution_block(initial_neurons * 4, conv2, kernel_initializer)

    # 16 -> 8
    conv4, corr_conv4 = convolution_block(initial_neurons * 8, conv3, kernel_initializer)

    # Middle block
    conv_middle = middle_convolution_block(initial_neurons *16, conv4, kernel_initializer)

    # 8 -> 16
    upconv4 = upconvolution_block(initial_neurons * 8, conv_middle, corr_conv4, kernel_initializer)

    # 16 -> 32
    upconv3 = upconvolution_block(initial_neurons * 4, upconv4, corr_conv3, kernel_initializer)

    # 32 -> 64
    upconv2 = upconvolution_block(initial_neurons * 2, upconv3, corr_conv2, kernel_initializer)

    # 64 -> 128
    upconv1 = upconvolution_block(initial_neurons, upconv2, corr_conv1, kernel_initializer)

    # Output layer of 128x128x1 image tensors
    output_layer = Conv2D(1, kernel_size=(1, 1), padding='same', activation='sigmoid')(upconv1)

    return output_layer


def fit_model(x_train, y_train, x_valid, y_valid, weights_filepath):
    """
    Compile the model and fit it.
    """
    input_layer = Input(shape=(128, 128, 1), batch_shape=None)
    output_layer = build_network(initial_neurons=16, input_layer=input_layer)
    model = Model(input_layer, output_layer)

    # Compile the model
    model.compile(optimizer=adam(lr=0.01),  # TODO SGD(lr=0.01, momentum=0.99, decay=0.)
                  loss='binary_crossentropy',
                  metrics=['accuracy', competition_metric])

    # Define some callbacks
    early_stopping = EarlyStopping(monitor='competition_metric', patience=10, verbose=1, mode='max')
    model_checkpoint = ModelCheckpoint(weights_filepath, monitor='competition_metric', verbose=1,
                                       save_best_only=True, save_weights_only=False, mode='max')
    reduce_rate = ReduceLROnPlateau(monitor='competition_metric', factor=0.5, patience=5, verbose=1,
                                    mode='max', min_lr=0.0001)

    # Choose which callbacks to include
    callbacks = [
        early_stopping,
        model_checkpoint,
        reduce_rate
    ]

    # Fit the model
    model.fit(x=x_train, y=y_train, batch_size=32, epochs=50, verbose=1,
              callbacks=callbacks, validation_data=[x_valid, y_valid])
    return model
