"""
Implements the pre-trained ResNet34 model.

Author: Sam Waterbury
GitHub: https://github.com/samwaterbury/salt-identification
"""

import numpy as np
import pandas as pd

from skimage.transform import resize

from keras.models import Model, load_model
from keras.layers import Input, Activation, BatchNormalization, Concatenate
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import add
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import adam, sgd
from keras.regularizers import l2
from keras import backend

from utilities import iou, iou_no_sigmoid, lovasz_loss, get_optimal_cutoff
from models.AbstractModel import AbstractModel


class ResNet34(AbstractModel):
    def __init__(self, config):
        """
        Initialize the model with the correct parameters and attributes.

        :param config: Config file dictionary.
        """
        self.model_name = 'ResNet34'
        self.img_size = 128
        self.custom_objects = {}

        self.parameters = config['model_parameters'][self.model_name]
        self.save_path = config['paths']['saved_model'].format(self.model_name)

        self.model = None
        self.optimal_cutoff = self.parameters['optimal_cutoff']
        self.is_fitted = False

    def load(self, save_path=None):
        """
        Load a saved ResNet34 and set it as the `model` self attribute.

        :param save_path: Path to saved mode. If not supplied, use default path.
        """
        save_path = self.save_path if save_path is None else save_path
        print('Loading saved {} model from {}...'.format(self.model_name, save_path))
        self.model = load_model(save_path, custom_objects=self.custom_objects)
        self.optimal_cutoff = self.parameters['optimal_cutoff']

    def train(self, x_train, y_train, x_valid, y_valid):
        """
        Train the model. If validation data is supplied, use it to prevent
        overfitting and estimate the optimal cutoff for making predictions.

        :param x_train: Series or list of numpy arrays containing images.
        :param y_train: Series or list of numpy arrays containing masks.
        :param x_valid: Series or list of numpy arrays containing images.
        :param y_valid: Series or list of numpy arrays containing masks.
        """
        print('Fitting {} (stage 1)...'.format(self.model_name))
        print('Fitting {} (stage 1)...'.format(self.model_name))
        if x_valid is None and y_valid is None:
            valid = None
        else:
            valid = [x_valid, y_valid]
            x_valid = self.preprocess(x_valid)
            y_train = self.preprocess(y_train)
        x_train = np.append(x_train, np.asarray([np.fliplr(image) for image in x_train]), axis=0)
        y_train = np.append(y_train, np.asarray([np.fliplr(image) for image in y_train]), axis=0)
        x_train = self.preprocess(x_train)
        y_train = self.preprocess(y_train)

        if self.parameters['optim_1'] == 'adam':
            optim1 = adam(lr=self.parameters['lr_1'])
        elif self.parameters['optim_1'] == 'sgd':
            optim1 = sgd(lr=self.parameters['lr_1'], momentum=self.parameters['optim_momentum_1'])
        else:
            raise NotImplementedError('Optimizer should be either adam or SGD.')
        self.model.compile(optimizer=optim1, loss='binary_crossentropy', metrics=[iou])

        callbacks = [
            EarlyStopping(monitor='iou', patience=self.parameters['early_stopping'], verbose=2),
            ReduceLROnPlateau(monitor='iou', factor=0.1, patience=4, verbose=2, min_lr=0.00001),
            ModelCheckpoint(self.save_path, monitor='iou', verbose=2, save_best_only=True)
        ]

        self.model.fit(x=x_train, y=y_train, batch_size=self.parameters['batch_size_1'],
                       epochs=self.parameters['epochs_1'], verbose=2, callbacks=callbacks,
                       validation_data=valid)
        load_model(self.save_path)

        print('Fitting {} (stage 2)...'.format(self.model_name))
        self.model = Model(self.model.layers[0].input, self.model.layers[-1].input)

        if self.parameters['optim_2'] == 'adam':
            optim2 = adam(lr=self.parameters['lr_2'])
        elif self.parameters['optim_2'] == 'sgd':
            optim2 = sgd(lr=self.parameters['lr_2'], momentum=self.parameters['optim_momentum_2'])
        else:
            raise NotImplementedError('Optimizer should be either adam or SGD.')
        self.model.compile(optimizer=optim2, loss=lovasz_loss, metrics=[iou_no_sigmoid])

        for cb in callbacks:
            cb.monitor = 'iou_no_sigmoid'

        self.model.fit(x=x_train, y=y_train, batch_size=self.parameters['batch_size_stage2'],
                       epochs=self.parameters['epochs_stage2'], verbose=2, callbacks=callbacks,
                       alidation_data=valid)
        self.load(self.save_path)

        # Determine the optimal likelihood cutoff for segmenting images
        if valid is not None:
            valid_predictions = self.predict(x_valid)
            valid_predictions = self.postprocess(valid_predictions)
            self.optimal_cutoff = get_optimal_cutoff(valid_predictions, y_valid)

    def predict(self, x):
        """
        Make predictions for a set of images.

        :param x: Array of images with shape (n, 101, 101, 1).
        :return: Array of pixel "probabilities" with shape (n, 101, 101).
        """
        print('Making predictions with {}...'.format(self.model_name))
        x_flip = np.array([np.fliplr(image) for image in x])
        predictions = self.model.predict(x).reshape(-1, self.img_size, self.img_size)
        predictions_mirrored = self.model.predict(x_flip).reshape(-1, self.img_size, self.img_size)
        predictions += np.array([np.fliplr(image) for image in predictions_mirrored])
        predictions = predictions / 2
        return predictions

    def preprocess(self, x):
        """
        Flatten a list of arrays so that it can be processed by the network.

        :param x: List or Series of 101x101 numpy arrays.
        :return: Numpy array with shape (n, 128, 128, 1).
        """
        def upsize(img):
            return resize(img, (self.img_size, self.img_size), mode='constant', preserve_range=True)
        if isinstance(x, pd.Series):
            x = x.map(upsize).tolist()
        x = list(map(upsize, x))
        return np.array(x).reshape(-1, self.img_size, self.img_size, 1)

    def postprocess(self, x):
        """
        No postprocessing needed for this model.
        """
        def downsize(img):
            return resize(img, (101, 101), mode='constant', preserve_range=True)
        np.apply_along_axis(downsize, axis=0, arr=x)
        return x

    def build(self):
        """
        Construct the entire network and save it as the `model` self attribute.
        """
        base_network = ResNet34.build_base_network(self.img_size)
        input_layer = base_network.input

        shortcuts = list([97, 54, 25])
        layer = base_network.output
        for i in range(5):
            shortcut = base_network.layers[shortcuts[i]].output if i < len(shortcuts) else None
            filters = 16 * 2 ** (6 - i)
            layer = ResNet34.upsample_block(layer, filters, False, shortcut)

        layer = Conv2D(filters=1, kernel_size=(3, 3), padding='same')(layer)
        output_layer = Activation('sigmoid')(layer)
        self.model = Model(input_layer, output_layer)

    @staticmethod
    def upsample_block(block_input, filters, batch_normalization=False, shortcut=None):
        """
        Return a function of layers which constitute an expansion block.

        :param block_input: Keras layer.
        :param filters: Number of filters/neurons for convolution.
        :param batch_normalization: Boolean indicating whether to use batch normalization.
        :param shortcut: Optional layer to concatenate with the input.
        :return: Function which takes in an input and applies multiple layers.
        """
        layer = UpSampling2D(size=(2, 2))(block_input)
        if shortcut is not None:
            layer = Concatenate()([layer, shortcut])
        layer = Conv2D(filters, (3, 3), padding='same')(layer)
        if batch_normalization:
            layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = Conv2D(filters, (3, 3), padding='same')(layer)
        if batch_normalization:
            layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        return layer

    @staticmethod
    def build_base_network(img_size):
        """
        Constructs the base U-Net model which the ResNet34 is built on top of.

        :param img_size: Width of the image arrays.
        :return: Network layers as a keras Model object.
        """
        input_layer = Input(shape=(img_size, img_size, 1))
        block = ResNet34.convolution_activation(input_layer, filters=64)
        block = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(block)

        filters = 64
        for i, r in enumerate([3, 4, 6, 3]):
            first_layer = True if i == 0 else False
            block = ResNet34.residual_block(block, filters, r, first_layer)
            filters *= 2
        output_layer = ResNet34.batch_activation(block)
        return Model(input_layer, output_layer)

    @staticmethod
    def batch_activation(input_layer):
        layer = BatchNormalization()(input_layer)
        layer = Activation('relu')(layer)
        return layer

    @staticmethod
    def convolution_activation(block_input, filters):
        """
        Convolution -> Batch normalization -> RELU.

        :param block_input: Keras layer.
        :param filters: Number of filters/neurons for convolution.
        :return: Keras layer.
        """

        layer = Conv2D(filters, kernel_size=(7, 7), strides=(2, 2), padding='same',
                       kernel_initializer='he_normal', kernel_regularizer=l2(0.0001))(block_input)
        layer = ResNet34.batch_activation(layer)
        return layer

    @staticmethod
    def batch_convolution(block_input, filters, strides=(1, 1)):
        """
        Batch normalization -> RELU -> Convolution.

        :param block_input: Keras layer.
        :param filters: Number of filters/neurons for convolution.
        :param strides: Tuple of integers (k, k) for downsampling the image.
        :return: Keras layer.
        """
        layer = ResNet34.batch_activation(block_input)
        layer = Conv2D(filters, kernel_size=(3, 3), strides=strides, padding='same',
                       kernel_initializer='he_normal', kernel_regularizer=l2(0.0001))(layer)
        return layer

    @staticmethod
    def shortcut(input_layer, residual):
        """
        Adds a layer from the downhill phase of the U-Net to a residual.

        :param input_layer: Unconvoluted keras Layer from the downhill phase.
        :param residual: Convoluted residual keras Layer.
        :return: Keras layer.
        """
        input_shape = backend.int_shape(input_layer)
        residual_shape = backend.int_shape(residual)
        stride_width = int(round(input_shape[1] / residual_shape[1]))
        stride_height = int(round(input_shape[2] / residual_shape[2]))
        equal_channels = input_shape[3] == residual_shape[3]

        shortcut = input_layer
        if stride_width > 1 or stride_height > 1 or not equal_channels:
            shortcut = Conv2D(filters=residual_shape[3], kernel_size=(1, 1),
                              strides=(stride_width, stride_height), padding='valid',
                              kernel_initializer='he_normal',
                              kernel_regularizer=l2(0.0001))(input_layer)
        return add([shortcut, residual])

    @staticmethod
    def residual_block(input_block, filters, repetitions, first_layer=False):
        layer = input_block
        for i in range(repetitions):
            init_strides = (2, 2) if i == 0 and not first_layer else (1, 1)
            if first_layer and i == 0:
                residual = Conv2D(filters, kernel_size=(3, 3), strides=init_strides, padding='same',
                                  kernel_initializer='he_normal',
                                  kernel_regularizer=l2(0.0001))(layer)
            else:
                residual = ResNet34.batch_convolution(layer, filters=filters, strides=init_strides)
            residual = ResNet34.batch_convolution(residual, filters=filters)
            layer = ResNet34.shortcut(layer, residual)

        return layer
