"""Implements the convolutional network for this project.

For more information about the architecture of this network, see the following
in the `papers/` folder:

- Huang et al. (2018): in-depth explanation of residual neural networks
- Ronneberger et al. (2015): see page 2 for a visualization of a "U-Net"

Author: Sam Waterbury
GitHub: https://github.com/samwaterbury/salt-identification
"""

import os
import warnings

import numpy as np
import pandas as pd

from keras.models import Model, load_model
from keras.layers import Activation, Add, BatchNormalization, Dropout, Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.merge import concatenate
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import adam

from src.utilities import DEFAULT_PATHS
from src.utilities import iou_sigmoid, iou_no_sigmoid, lovasz_loss, get_cutoff


DEFAULT_MODEL_PARAMETERS = {
    'model_name': 'CustomResNet',
    'optimal_cutoff': -0.15,  # Found by testing; approx ~= ln(0.46 / 0.54)
    'initial_neurons': 16,
    'kernel_initializer': 'he_normal',
    'dropout_rate': 0.5,
    'batchnorm_momentum': 0.99,
    'early_stopping_patience': 50,
    'batch_size': 32,
    'round1_lr': 0.01,
    'round2_lr': 0.0005,
    'round3_lr': 0.0001,
    'round1_epochs': 10,  # This is way too few epochs to get good results, but
    'round2_epochs': 10,  # it will run in a reasonable amount of time and
    'round3_epochs': 10   # still give a good idea of how it works.
}


class CustomResNet(object):
    """Custom convolutional network with a residual "U-Net" architecture.

    This class contains a custom model used for the TGS competition. It takes
    in image arrays of shape (101, 101, 1), i.e., single channel. Methods are
    provided for preprocessing, training the model, and making predictions.

    `CustomResNet` relies on a set of parameters which can be passed when the
    class is initialized or whenever a method is called.

    Parameters (Keyword Args):
        model_name: Name for the model; used when saving model weights.
        optimal_cutoff: Cutoff value to use when making predictions.
        initial_neurons: # of filters at the first convolution output.
        kernel_initializer: 'he_normal', 'glorot_normal', etc.
        dropout_rate: % of nodes to randomly drop at each dropout layer.
        batchnorm_momentum: "Momentum" parameter for batch normalization.
        early_stopping_patience: Number of epochs without improvement to
            wait before ending the fitting process.
        batch_size: Batch size for training.
        lr1: round 1 (binary crossentropy loss) learning rate.
        lr2: round 2 (lovasz loss) learning rate.
        lr3: round 3 (lovasz loss) learning rate.
        round1_epochs: Round 1 (binary crossentropy loss) learning rate.
        round2_epochs: Round 2 (lovasz loss) learning rate.
        round3_epochs: Round 3 (lovasz loss) learning rate.

    Attributes:
        parameters: Dictionary containing all of the model parameters.
        custom_objects: Custom functions used during the training process.
        model: Keras Model class containing the network layers.
        save_path: Save location of model weights.
        stage: Value representing the current state of the model.
            0 - Model has been built but not fitted.
            1 - Model has been fitted but still has the sigmoid output layer.
            2 - Model has been fitted and the sigmoid output layer is removed.
        optimal_cutoff: The best cutoff value to use when rounding predictions.
    """

    def __init__(self, **kwargs):
        self.parameters = DEFAULT_MODEL_PARAMETERS
        self.parameters.update(kwargs)
        self.custom_objects = {
            'lovasz_loss': lovasz_loss,
            'iou_no_sigmoid': iou_no_sigmoid,
            'iou_sigmoid': iou_sigmoid
        }

        self.model = self.build()
        self.save_path = os.path.join(DEFAULT_PATHS['dir_output'],
                                      self.parameters['model_name'] + '.model')
        self._is_fitted = False

    @property
    def stage(self):
        """Value representing the current model state (see class docstring)."""
        if not self._is_fitted:
            return 0
        if self.model.layers[-1].name == 'sigmoid':
            return 1
        return 2

    @property
    def optimal_cutoff(self):
        """Provides cleaner external access to the optimal cutoff."""
        return self.parameters.get('optimal_cutoff', 0.)

    def load(self, save_path):
        """Loads a previously saved model.

        Args:
            save_path: Path to the saved model weights.
        """
        if not os.path.exists(save_path):
            raise FileNotFoundError(
                'Could not find the saved model at {}'.format(save_path))
        if self.stage <= 1:
            self.remove_sigmoid()
        self.model = load_model(save_path, custom_objects=self.custom_objects)
        self._is_fitted = True

    @staticmethod
    def process(x):
        """Flattens a list of arrays so that it can be used by the network.

        Args:
            x: List or Series of n arrays with shape (101, 101).

        Returns:
            Numpy array with shape (n, 101, 101, 1).
        """
        if isinstance(x, pd.Series):
            x = x.tolist()
        return np.reshape(np.asarray(x), newshape=(-1, 101, 101, 1))

    def train(self, x_train, y_train, x_valid=None, y_valid=None, **kwargs):
        """Trains the model and finds the optimal cutoff for predictions.

        Args:
            x_train: List or array of training images.
            x_valid: List or array of validation images.
            y_train: List or array of training masks.
            y_valid: List of array of validation masks.

        Keyword Args:
            Any class or model parameters (see class docstring).
        """
        self.parameters.update(kwargs)
        print('Fitting model (round 1)...')

        if x_valid is not None and y_train is not None:
            validation_set = self.process(x_valid), self.process(y_valid)
        else:
            validation_set = None

        # Preprocess the training data and add horizontal flip augmentations
        x_train = self.process(x_train)
        y_train = self.process(y_train)
        x_train = np.append(x_train, np.fliplr(x_train), axis=0)
        y_train = np.append(y_train, np.fliplr(y_train), axis=0)

        # Compile the model for round 1
        round1_optim = adam(lr=self.parameters['round1_lr'])
        self.model.compile(optimizer=round1_optim, loss='binary_crossentropy',
                           metrics=[iou_sigmoid])

        callbacks = [
            EarlyStopping(monitor='iou_sigmoid', mode='max', verbose=2,
                          patience=self.parameters['early_stopping_patience']),
            ReduceLROnPlateau(monitor='iou_sigmoid', mode='max', factor=0.5,
                              patience=5, verbose=2, min_lr=0.0001),
            ModelCheckpoint(self.save_path, monitor='iou_sigmoid',
                            mode='max', verbose=2, save_best_only=True)
        ]

        self.model.fit(x=x_train, y=y_train,
                       validation_data=validation_set,
                       batch_size=self.parameters['batch_size'],
                       epochs=self.parameters['round1_epochs'],
                       callbacks=callbacks,
                       verbose=2)
        self.load(self.save_path)

        print('Fitting model (round 2)...')

        # Compile the model for round 2
        self.remove_sigmoid()
        round2_optim = adam(lr=self.parameters['round2_lr'])
        self.model.compile(optimizer=round2_optim, loss=lovasz_loss,
                           metrics=[iou_no_sigmoid])

        # Callbacks need to use the correct scoring function for the new loss
        for cb in callbacks:
            cb.monitor = 'iou_no_sigmoid'

        self.model.fit(x=x_train, y=y_train,
                       validation_data=validation_set,
                       batch_size=self.parameters['batch_size'],
                       epochs=self.parameters['round2_epochs'],
                       callbacks=callbacks,
                       verbose=2)
        self.load(self.save_path)

        print('Fitting model (round 3)...')

        # Compile the model for round 3
        round3_optim = adam(lr=self.parameters['round3_lr'])
        self.model.compile(optimizer=round3_optim, loss=lovasz_loss,
                           metrics=[iou_no_sigmoid])

        self.model.fit(x=x_train, y=y_train,
                       validation_data=validation_set,
                       batch_size=self.parameters['batch_size'],
                       epochs=self.parameters['round3_epochs'],
                       callbacks=callbacks,
                       verbose=2)
        self.load(self.save_path)

        # Determine the optimal likelihood cutoff for segmenting images
        if validation_set:
            valid_predictions = self.predict(validation_set[0])
            optimal_cutoff = get_cutoff(valid_predictions, validation_set[1])
            self.parameters['optimal_cutoff'] = optimal_cutoff

        self._is_fitted = True

    def predict(self, x):
        """Generates predictions for a set of images.

        Args:
            x: Array of images with shape (n, 101, 101, 1).

        Returns:
            Array of pixel "probabilities" with shape (n, 101, 101).
        """
        # Flip all of the images horizontally
        x = self.process(x)
        x_flip = np.fliplr(x)

        # Make predicitons on the original images and the flipped images
        predictions = self.model.predict(x).reshape(-1, 101, 101)
        predictions_flip = self.model.predict(x_flip).reshape(-1, 101, 101)
        predictions_flip = np.fliplr(predictions_flip)

        # Take the average
        predictions += predictions_flip
        predictions /= 2
        return predictions

    @staticmethod
    def build(initial_neurons=16, kernel_initializer='he_normal',
              dropout_rate=0.5, bn_momentum=0.99):
        """Constructs the entire network in preparation for training/loading.

        Args:
            initial_neurons: Number of filters at the first convolution.
            kernel_initializer: Method of initializing the kernel weights. The
                original paper recommends "he_normal" but "glorot_normal" works
                well too.
            dropout_rate: % of filters to drop in dropout layers. The standard
                is 0.5, but a lower value might work better in this model.
            bn_momentum: Momentum for moving mean/variance in the batch
                normalization blocks.
        """
        # Line space is precious!
        n_init = initial_neurons
        ki = kernel_initializer
        dr = dropout_rate
        bn = bn_momentum

        # The network takes in images with the shape (101, 101, 1)
        input_layer = Input(shape=(101, 101, 1))

        # Coming up next is the U-Net architecture, which consists of a
        # "downslope" phase, "middle" phase, and "upslope" phase. The following
        # sequence of transformations are repeatedly applied on the downslope:
        #
        #   (1) Convolution
        #   (2) Two residual blocks (see the static method `residual_block`)
        #   (3) RELU activation function
        #   (4) Pooling function (this reduces the size of the image)
        #   (5) Dropout layer
        #
        # In the middle, these transformations are performed another time, but
        # the pooling and dropout layers are skipped. Then, the following
        # sequence of transformations are repeatedly applied on the upslope:
        #
        #   (1) Deconvolution (this increases the size of the image)
        #   (2) Concatenation with the corresponding layer from the downslope
        #   (3) Dropout layer
        #   (4) Convolution
        #   (5) Two residual blocks
        #   (6) RELU activation function
        #
        # For more info, see Ronneberger et al. (2015) in the `papers/` folder.

        # Downslope portion of the U-Net begins here

        # 101 -> 50
        c1 = Conv2D(n_init * 1, (3, 3), padding='same',
                    kernel_initializer=ki)(input_layer)
        c1 = CustomResNet.residual_block(c1, n_init * 1, False, bn)
        c1 = CustomResNet.residual_block(c1, n_init * 1, True, bn)
        p1 = MaxPooling2D(pool_size=(2, 2))(c1)
        p1 = Dropout(rate=dr)(p1)

        # 50 -> 25
        c2 = Conv2D(n_init * 2, (3, 3), padding='same',
                    kernel_initializer=ki)(p1)
        c2 = CustomResNet.residual_block(c2, n_init * 2, False, bn)
        c2 = CustomResNet.residual_block(c2, n_init * 2, True, bn)
        p2 = MaxPooling2D(pool_size=(2, 2))(c2)
        p2 = Dropout(rate=dr)(p2)

        # 25 -> 12
        c3 = Conv2D(n_init * 4, (3, 3), padding='same',
                    kernel_initializer=ki)(p2)
        c3 = CustomResNet.residual_block(c3, n_init * 4, False, bn)
        c3 = CustomResNet.residual_block(c3, n_init * 4, True, bn)
        p3 = MaxPooling2D(pool_size=(2, 2))(c3)
        p3 = Dropout(rate=dr)(p3)

        # 12 -> 6
        c4 = Conv2D(n_init * 8, (3, 3), padding='same',
                    kernel_initializer=ki)(p3)
        c4 = CustomResNet.residual_block(c4, n_init * 8, False, bn)
        c4 = CustomResNet.residual_block(c4, n_init * 8, True, bn)
        p4 = MaxPooling2D(pool_size=(2, 2))(c4)
        p4 = Dropout(rate=dr)(p4)

        # Middle portion of the U-Net; the image size does not change here
        cm = Conv2D(n_init * 16, (3, 3), padding='same',
                    kernel_initializer=ki)(p4)
        cm = CustomResNet.residual_block(cm, n_init * 16, False, bn)
        cm = CustomResNet.residual_block(cm, n_init * 16, True, bn)

        # Upslope portion of the U-Net begins here

        # 6 -> 12
        t4 = Conv2DTranspose(n_init * 8, (3, 3), strides=(2, 2),
                             padding='same', kernel_initializer=ki)(cm)
        u4 = concatenate([t4, c4])
        u4 = Dropout(rate=dr)(u4)

        u4 = Conv2D(n_init * 8, (3, 3), padding='same',
                    kernel_initializer=ki)(u4)
        u4 = CustomResNet.residual_block(u4, n_init * 8, False, bn)
        u4 = CustomResNet.residual_block(u4, n_init * 8, True, bn)

        # 12 -> 25
        t3 = Conv2DTranspose(n_init * 4, (3, 3), strides=(2, 2),
                             padding='valid', kernel_initializer=ki)(u4)
        u3 = concatenate([t3, c3])
        u3 = Dropout(rate=dr)(u3)

        u3 = Conv2D(n_init * 4, (3, 3), padding='same',
                    kernel_initializer=ki)(u3)
        u3 = CustomResNet.residual_block(u3, n_init * 4, False, bn)
        u3 = CustomResNet.residual_block(u3, n_init * 4, True, bn)

        # 25 -> 50
        t2 = Conv2DTranspose(n_init * 2, (3, 3), strides=(2, 2),
                             padding='same', kernel_initializer=ki)(u3)
        u2 = concatenate([t2, c2])
        u2 = Dropout(rate=dr)(u2)

        u2 = Conv2D(n_init * 2, (3, 3), padding='same',
                    kernel_initializer=ki)(u2)
        u2 = CustomResNet.residual_block(u2, n_init * 2, False, bn)
        u2 = CustomResNet.residual_block(u2, n_init * 2, True, bn)

        # 50 -> 101
        t1 = Conv2DTranspose(n_init * 1, (3, 3), strides=(2, 2),
                             padding='valid', kernel_initializer=ki)(u2)
        u1 = concatenate([t1, c1])
        u1 = Dropout(rate=dr)(u1)

        u1 = Conv2D(n_init * 1, (3, 3), padding='same',
                    kernel_initializer=ki)(u1)
        u1 = CustomResNet.residual_block(u1, n_init * 1, False, bn)
        u1 = CustomResNet.residual_block(u1, n_init * 1, True, bn)

        # Output layer
        output_layer = Conv2D(1, (1, 1), padding='same',
                              kernel_initializer=ki)(u1)
        output_layer = Activation('sigmoid', name='sigmoid')(output_layer)

        return Model(input_layer, output_layer)

    def remove_sigmoid(self):
        """Removes the sigmoid activation in the output layer of the network.

        This network is trained in three rounds, using a different loss
        function in the latter two. After the first round, the sigmoid
        activation in the output layer must be removed because the second loss
        function (Lovasz) requires values in (-Inf, Inf) instead of [0, 1].
        """
        if self.stage > 1:
            warnings.warn('The sigmoid output layer has already been removed.')
            return
        input_layer = self.model.layers[0].input
        output_layer = self.model.layers[-1].input
        self.model = Model(input_layer, output_layer)

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
        layer = CustomResNet.convolution_block(layer, filters,
                                               batch_relu=False,
                                               bn_momentum=bn_momentum)
        layer = Add()([layer, block_input])
        if use_activation:
            layer = CustomResNet.batch_activation(layer, bn_momentum)
        return layer
