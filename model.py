"""
This file constructs the model upon a call to `get_fitted_model()`.

Author: Sam Waterbury
GitHub: https://github.com/samwaterbury/salt-identification
"""

from keras.optimizers import SGD
from keras.models import Model, load_model
from keras.layers import Input, Dropout, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint

from utilities import iou_bce_loss, competition_metric


def convolution_block(neurons, block_input):
    conv1 = Conv2D(neurons, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')(block_input)
    conv1 = Activation('relu')(conv1)
    conv2 = Conv2D(neurons, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')(conv1)
    conv2 = Activation('relu')(conv2)
    pool = MaxPooling2D(pool_size=(2, 2), padding='valid')(conv2)
    return pool, conv2


def middle_convolution_block(neurons, block_input):
    conv1 = Conv2D(neurons, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')(block_input)
    conv1 = Activation('relu')(conv1)
    conv2 = Conv2D(neurons, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')(conv1)
    conv2 = Activation('relu')(conv2)
    return conv2


def upconvolution_block(neurons, block_input, corr_conv):
    deconv = Conv2DTranspose(neurons, kernel_size=(3, 3), strides=(2, 2), padding='same')(block_input)
    upconv = concatenate([deconv, corr_conv])
    upconv = Conv2D(neurons, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')(upconv)
    upconv = Activation('relu')(upconv)
    upconv = Conv2D(neurons, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')(upconv)
    upconv = Activation('relu')(upconv)
    return upconv


def build_network(initial_neurons):

    # Input layer of 128x128x1 image tensors
    input_layer = Input(shape=(128, 128, 1), batch_shape=None)

    # 128 -> 64
    conv1, corr_conv1 = convolution_block(initial_neurons, input_layer)

    # 64 -> 32
    conv2, corr_conv2 = convolution_block(initial_neurons * 2, conv1)

    # 32 -> 16
    conv3, corr_conv3 = convolution_block(initial_neurons * 4, conv2)

    # 16 -> 8
    conv4, corr_conv4 = convolution_block(initial_neurons * 8, conv3)

    # Middle block
    conv_middle = middle_convolution_block(initial_neurons *16, conv4)

    # 8 -> 16
    upconv4 = upconvolution_block(initial_neurons * 8, conv_middle, corr_conv4)

    # 16 -> 32
    upconv3 = upconvolution_block(initial_neurons * 4, upconv4, corr_conv3)

    # 32 -> 64
    upconv2 = upconvolution_block(initial_neurons * 2, upconv3, corr_conv2)

    # 64 -> 128
    upconv1 = upconvolution_block(initial_neurons, upconv2, corr_conv1)

    # Output layer of 128x128x1 image tensors
    output_layer = Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation='sigmoid')(upconv1)

    network = Model(input_layer, output_layer)
    network.compile(
        optimizer=SGD(lr=0.01, momentum=0.99, decay=0.),
        loss=iou_bce_loss,
        metrics=['accuracy', competition_metric]
    )
    return network


def fit_model(X_train, y_train, X_valid, y_valid, weights_filepath):
    model = build_network(initial_neurons=8)
    callbacks = [
        EarlyStopping(patience=10, verbose=1),
        ModelCheckpoint(weights_filepath, save_best_only=True, save_weights_only=True, mode='min')
    ]
    model.fit(x=X_train, y=y_train, batch_size=1, epochs=1, verbose=1,
              callbacks=callbacks, validation_data=[X_valid, y_valid])
    return model
