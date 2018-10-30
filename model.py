"""
Implements the custom residual network with a U-Net architecture. For more
information about this architecture, see the following in the `papers/` folder:

- Huang et al. (2018): in-depth explanation of residual neural networks
- Ronneberger et al. (2015): see page 2 for a visualization of a "U-Net"

Author: Sam Waterbury
GitHub: https://github.com/samwaterbury/salt-identification
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd

from keras.models import Model, load_model
from keras.layers import Activation, Add, BatchNormalization, Dropout, Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.merge import concatenate
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import adam

from utilities import iou_sigmoid, iou_no_sigmoid, lovasz_loss, get_cutoff


class CustomResNet(object):
    """Fully-contained residual network with a easy-to-use interface.

    This class contains a custom model used for the TGS competition. It takes in
    image arrays of shape (101, 101, 1), i.e., single channel. Methods are
    provided for preprocessing, training the model, and making predictions.

    `CustomResNet` relies on a set of parameters which can be set when the class
    is initialized or whenever a function is called. For a list and descriptions
    of these parameters, please see the method `set_parameters`.

    Attributes:
        name: Name of this model instance.
        model: Keras `Model` object.
        is_fitted: True if the model has been fitted, False otherwise.
        optimal_cutoff: Cutoff value to use when making predictions.
        stage: 0 if the model has not been built, 1 if the model is built and
            still contains the sigmoid output layer, 2 if the sigmoid output
            layer has been removed.
        custom_objects: Custom functions used during the training process.
        parameters: Dictionary containing all of the model parameters.
    """
    def __init__(self, name=None, output=None, **kwargs):
        """Initializes the model with the correct parameters and attributes.

        Args:
            name: Optional, used to save model weights.
            output: Directory to save all output to (e.g., saved model weights).
            **kwargs: See `set_parameters` for a list of parameters you can set.
        """
        self.name = name if name else 'CustomResNet-{}'.format(
            datetime.now().strftime('%Y%m%d_%I%M%p'))
        self.output_dir = output
        self.model = None
        self.is_fitted = False
        self.stage = 0
        self.custom_objects = {
            'lovasz_loss': lovasz_loss,
            'iou_no_sigmoid': iou_no_sigmoid,
            'iou_sigmoid': iou_sigmoid
        }

        self.parameters = {}
        self.set_parameters(**kwargs)

        self.build()
        if self.parameters['load_model'] and self.stage > 0:
            self.load(self.parameters['save_path'])

    @property
    def optimal_cutoff(self):
        """Provides cleaner external access to the optimal cutoff."""
        return self.parameters.get('optimal_cutoff', 0.)

    def set_parameters(self, **kwargs):
        """Updates the parameters for the model.

        Keyword Args:
            load_model: If True, load the model weights from `save_path`.
            save_path: Path to the saved model weights.
            optimal_cutoff: Cutoff value to use when making predictions.
            initial_neurons: # of filters at the first convolution output.
            kernel_initializer: 'he_normal', 'glorot_normal', etc.
            dropout_rate: % of nodes to randomly drop at each dropout layer.
            batchnorm_momentum: "Momentum" parameter for batch normalization.
            early_stopping_patience: Number of epochs without improvement to
                wait before ending the fitting process.
            batch_size: Batch size for training.
            lr1: Stage 1 (binary crossentropy loss) learning rate.
            lr2: Stage 2 (lovasz loss) learning rate.
            lr3: Stage 3 (lovasz loss) learning rate.
            stage1_epochs: Stage 1 (binary crossentropy loss) learning rate.
            stage2_epochs: Stage 2 (lovasz loss) learning rate.
            stage3_epochs: Stage 3 (lovasz loss) learning rate.
        """
        _parameters = {
            'load_model': False,
            'save_name': os.path.join(self.output_dir,
                                      'saved_{}.model'.format(self.name)),
            'optimal_cutoff': 0.,
            'initial_neurons': 16,
            'kernel_initializer': 'he_normal',
            'dropout_rate': 0.5,
            'batchnorm_momentum': 0.99,
            'early_stopping_patience': 50,
            'batch_size': 32,
            'stage1_lr': 0.01,
            'stage2_lr': 0.0005,
            'stage3_lr': 0.0001,
            'stage1_epochs': 10,  # This is too few epochs to get good results,
            'stage2_epochs': 10,  # but it will run in a reasonable amount of
            'stage3_epochs': 10   # time and give a good idea of how it works.
        }
        _parameters.update(self.parameters)
        _parameters.update(kwargs)
        self.parameters = _parameters

    def load(self, save_path=None):
        """Loads a saved model.

        Args:
            save_path: If left as None, the save path parameter will be used.
        """
        if save_path is None:
            save_path = self.parameters['save_path']
        if not os.path.exists(save_path):
            raise FileNotFoundError('Could not find the saved model at {}'
                                    .format(save_path))
        if self.stage <= 1:
            self.remove_sigmoid()
        self.model = load_model(save_path, custom_objects=self.custom_objects)
        self.is_fitted = True

    @staticmethod
    def process(x):
        """Flattens a list of arrays so that it can be processed by the network.

        Args:
            x: List or Series of n arrays with shape (101, 101).

        Returns:
            Numpy array with shape (n, 101, 101, 1).
        """
        if isinstance(x, pd.Series):
            x = x.tolist()
        return np.array(x).reshape(-1, 101, 101, 1)

    def train(self, x_train, y_train, x_valid=None, y_valid=None):
        """Trains the model and finds the optimal cutoff for predictions.

        Args:
            x_train: List or array of training images.
            x_valid: List or array of validation images.
            y_train: List or array of training masks.
            y_valid: List of array of validation masks.
        """
        print('Fitting model (stage 1)...')

        if x_valid is not None and y_train is not None:
            valid = [self.process(x_valid), self.process(y_valid)]
        else:
            valid = None

        # Preprocess the training data and add horizontal flip augmentations
        x_train = self.process(x_train)
        y_train = self.process(y_train)
        x_train = np.append(x_train, [np.fliplr(i) for i in x_train], axis=0)
        y_train = np.append(y_train, [np.fliplr(i) for i in y_train], axis=0)

        # Compile the model for stage 1
        stage1_optim = adam(lr=self.parameters['stage1_lr'])
        self.model.compile(optimizer=stage1_optim, loss='binary_crossentropy',
                           metrics=[iou_sigmoid])

        callbacks = [
            EarlyStopping(monitor='iou_sigmoid', mode='max', verbose=2,
                          patience=self.parameters['early_stopping_patience']),
            ReduceLROnPlateau(monitor='iou_sigmoid', mode='max', factor=0.5,
                              patience=5, verbose=2, min_lr=0.0001),
            ModelCheckpoint(self.parameters['save_path'], monitor='iou_sigmoid',
                            mode='max', verbose=2, save_best_only=True)
        ]

        self.model.fit(x=x_train, y=y_train, validation_data=valid,
                       batch_size=self.parameters['batch_size'],
                       epochs=self.parameters['stage1_epochs'],
                       callbacks=callbacks,
                       verbose=2)
        self.load(self.parameters['save_path'])

        print('Fitting model (stage 2)...')

        # Compile the model for stage 2
        self.remove_sigmoid()
        stage2_optim = adam(lr=self.parameters['stage2_lr'])
        self.model.compile(optimizer=stage2_optim, loss=lovasz_loss,
                           metrics=[iou_no_sigmoid])

        # Callbacks need to use the correct scoring function for the new loss
        for cb in callbacks:
            cb.monitor = 'iou_no_sigmoid'

        self.model.fit(x=x_train, y=y_train, validation_data=valid,
                       batch_size=self.parameters['batch_size'],
                       epochs=self.parameters['stage2_epochs'],
                       callbacks=callbacks,
                       verbose=2)
        self.load(self.parameters['save_path'])

        print('Fitting model (stage 3)...')

        # Compile the model for stage 3
        stage3_optim = adam(lr=self.parameters['stage3_lr'])
        self.model.compile(optimizer=stage3_optim, loss=lovasz_loss,
                           metrics=[iou_no_sigmoid])

        self.model.fit(x=x_train, y=y_train, validation_data=valid,
                       batch_size=self.parameters['batch_size'],
                       epochs=self.parameters['stage3_epochs'],
                       callbacks=callbacks,
                       verbose=2)
        self.load(self.parameters['save_path'])
        self.stage = 3

        # Determine the optimal likelihood cutoff for segmenting images
        if valid:
            valid_predictions = self.predict(x_valid)
            optimal_cutoff = get_cutoff(valid_predictions, y_valid)
            print('Found optimal cutoff for {}: {}'.format(self.name,
                                                           optimal_cutoff))
            if 'optimal_cutoff' not in self.parameters:
                self.parameters['optimal_cutoff'] = optimal_cutoff

    def predict(self, x):
        """Generates predictions for a set of images.

        Args:
            x: Array of images with shape (n, 101, 101, 1).

        Returns:
            Array of pixel "probabilities" with shape (n, 101, 101).
        """
        # Flip all of the images horizontally
        x = self.process(x)
        x_flip = self.process([np.fliplr(i) for i in x])

        # Make predicitons on the original images and the flipped images
        predictions = self.model.predict(x).reshape(-1, 101, 101)
        predictions_flip = self.model.predict(x_flip).reshape(-1, 101, 101)
        predictions_flip = [np.fliplr(i) for i in predictions_flip]
        predictions_flip = np.array(predictions_flip).reshape(-1, 101, 101)

        # Take the average
        predictions += predictions_flip
        predictions /= 2
        return predictions

    def build(self):
        """Constructs the entire network in preparation for training/loading."""
        n_init = self.parameters['initial_neurons']
        ki = self.parameters['kernel_initializer']
        do = self.parameters['dropout_rate']
        bn = self.parameters['batchnorm_momentum']

        # The network takes in images with the shape (101, 101, 1)
        input_layer = Input(shape=(101, 101, 1))

        # Coming up next is the U-Net architecture, which repeatedly applies the
        # following sequence of transformations:
        #   (1) Convolution
        #   (2) Residual blocks (see the static method `residual_block`)
        #   (3) Pooling function (this reduces the size of the image)
        #   (4) Activation function
        #   (5) Dropout (optional)

        # 101 -> 50
        c1 = Conv2D(n_init * 1, (3, 3), padding='same',
                    kernel_initializer=ki)(input_layer)
        c1 = CustomResNet.residual_block(c1, n_init * 1, False, bn)
        c1 = CustomResNet.residual_block(c1, n_init * 1, True, bn)
        p1 = MaxPooling2D(pool_size=(2, 2))(c1)
        p1 = Dropout(rate=do)(p1)

        # 50 -> 25
        c2 = Conv2D(n_init * 2, (3, 3), padding='same',
                    kernel_initializer=ki)(p1)
        c2 = CustomResNet.residual_block(c2, n_init * 2, False, bn)
        c2 = CustomResNet.residual_block(c2, n_init * 2, True, bn)
        p2 = MaxPooling2D(pool_size=(2, 2))(c2)
        p2 = Dropout(rate=do)(p2)

        # 25 -> 12
        c3 = Conv2D(n_init * 4, (3, 3), padding='same',
                    kernel_initializer=ki)(p2)
        c3 = CustomResNet.residual_block(c3, n_init * 4, False, bn)
        c3 = CustomResNet.residual_block(c3, n_init * 4, True, bn)
        p3 = MaxPooling2D(pool_size=(2, 2))(c3)
        p3 = Dropout(rate=do)(p3)

        # 12 -> 6
        c4 = Conv2D(n_init * 8, (3, 3), padding='same',
                    kernel_initializer=ki)(p3)
        c4 = CustomResNet.residual_block(c4, n_init * 8, False, bn)
        c4 = CustomResNet.residual_block(c4, n_init * 8, True, bn)
        p4 = MaxPooling2D(pool_size=(2, 2))(c4)
        p4 = Dropout(rate=do)(p4)

        # Middle block skips the pooling function and does not use dropout
        cm = Conv2D(n_init * 16, (3, 3), padding='same',
                    kernel_initializer=ki)(p4)
        cm = CustomResNet.residual_block(cm, n_init * 16, False, bn)
        cm = CustomResNet.residual_block(cm, n_init * 16, True, bn)

        # From here on out, the image tensors are growing in size at each block

        # 6 -> 12
        d4 = Conv2DTranspose(n_init * 8, (3, 3), strides=(2, 2), padding='same',
                             kernel_initializer=ki)(cm)
        u4 = concatenate([d4, c4])
        u4 = Dropout(rate=do)(u4)

        u4 = Conv2D(n_init * 8, (3, 3), padding='same',
                    kernel_initializer=ki)(u4)
        u4 = CustomResNet.residual_block(u4, n_init * 8, False, bn)
        u4 = CustomResNet.residual_block(u4, n_init * 8, True, bn)

        # 12 -> 25
        d3 = Conv2DTranspose(n_init * 4, (3, 3), strides=(2, 2), padding='valid',
                             kernel_initializer=ki)(u4)
        u3 = concatenate([d3, c3])
        u3 = Dropout(rate=do)(u3)

        u3 = Conv2D(n_init * 4, (3, 3), padding='same',
                    kernel_initializer=ki)(u3)
        u3 = CustomResNet.residual_block(u3, n_init * 4, False, bn)
        u3 = CustomResNet.residual_block(u3, n_init * 4, True, bn)

        # 25 -> 50
        d2 = Conv2DTranspose(n_init * 2, (3, 3), strides=(2, 2), padding='same',
                             kernel_initializer=ki)(u3)
        u2 = concatenate([d2, c2])
        u2 = Dropout(rate=do)(u2)

        u2 = Conv2D(n_init * 2, (3, 3), padding='same',
                    kernel_initializer=ki)(u2)
        u2 = CustomResNet.residual_block(u2, n_init * 2, False, bn)
        u2 = CustomResNet.residual_block(u2, n_init * 2, True, bn)

        # 50 -> 101
        d1 = Conv2DTranspose(n_init * 1, (3, 3), strides=(2, 2),
                             padding='valid', kernel_initializer=ki)(u2)
        u1 = concatenate([d1, c1])
        u1 = Dropout(rate=do)(u1)

        u1 = Conv2D(n_init * 1, (3, 3), padding='same',
                    kernel_initializer=ki)(u1)
        u1 = CustomResNet.residual_block(u1, n_init * 1, False, bn)
        u1 = CustomResNet.residual_block(u1, n_init * 1, True, bn)

        # Output layer
        output_layer = Conv2D(1, (1, 1), padding='same',
                              kernel_initializer=ki)(u1)
        output_layer = Activation('sigmoid')(output_layer)

        self.model = Model(input_layer, output_layer)
        self.stage = 1

    def remove_sigmoid(self):
        """Removes the sigmoid activation in the output layer of the network.

        This network is trained in two stages, using a different loss function
        in each stage. In the second stage, the sigmoid activation in the output
        layer must be removed because the second loss function (Lovasz) requires
        values in (-Inf, Inf) instead of [0, 1].
        """
        if self.stage == 2:
            print('Model is already built for stage 2 of training.')
            return
        input_layer = self.model.layers[0].input
        output_layer = self.model.layers[-1].input
        self.model = Model(input_layer, output_layer)
        self.stage = 2

    @staticmethod
    def batch_activation(block_input, bn_momentum):
        """Applies batch normalization and RELU.

        Args:
            block_input: Keras layer.
            bn_momentum: Real number in range (0, 1); standard value is 0.99.

        Returns:
            Keras layer.
        """
        layer = BatchNormalization(momentum=bn_momentum)(block_input)
        layer = Activation('relu')(layer)
        return layer

    @staticmethod
    def convolution_block(block_input, filters, batch_relu=True,
                          bn_momentum=0.99):
        """Convolution followed by optional batch normalization and RELU.

        Args:
            block_input: Keras layer.
            filters: Number of filters/neurons for convolution.
            batch_relu: Boolean indicating whether to use BN and RELU.
            bn_momentum: Real number in range (0, 1); standard value is 0.99.

        Returns:
            Keras layer.
        """
        layer = Conv2D(filters, kernel_size=(3, 3), strides=(1, 1),
                       kernel_initializer='glorot_normal',
                       padding='same')(block_input)
        if batch_relu:
            layer = CustomResNet.batch_activation(layer, bn_momentum)
        return layer

    @staticmethod
    def residual_block(block_input, filters, use_activation=False,
                       bn_momentum=0.99):
        """RELU -> Convolution -> RELU -> Convolution -> Add -> Optional RELU

        Args:
            block_input: Keras layer.
            filters: Number of filters/neurons for convolution.
            use_activation: Boolean indicating whether to end with BN & RELU.
            bn_momentum: Real number in range (0, 1); standard value is 0.99.

        Returns:
            Keras layer.
        """
        layer = CustomResNet.batch_activation(block_input, bn_momentum)
        layer = CustomResNet.convolution_block(layer, filters,
                                               bn_momentum=bn_momentum)
        layer = CustomResNet.convolution_block(layer, filters, batch_relu=False,
                                               bn_momentum=bn_momentum)
        layer = Add()([layer, block_input])
        if use_activation:
            layer = CustomResNet.batch_activation(layer, bn_momentum)
        return layer
