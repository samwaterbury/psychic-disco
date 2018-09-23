"""
Implements the custom U-Net model with CustomResNet blocks.

Author: Sam Waterbury
GitHub: https://github.com/samwaterbury/salt-identification
"""

import numpy as np
import pandas as pd

from keras.models import Model, load_model
from keras.layers import Input, Add, Activation, BatchNormalization, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import adam, sgd

from utilities import iou, iou_no_sigmoid, lovasz_loss, get_optimal_cutoff
from models.AbstractModel import AbstractModel


class CustomResNet(AbstractModel):
    def __init__(self, config):
        self.model_name = 'CustomResNet'
        self.image_width = 101
        self.custom_objects = {
            'lovasz_loss': lovasz_loss,
            'iou_no_sigmoid': iou_no_sigmoid,
            'iou': iou
        }

        self.parameters = config['model_parameters'][self.model_name]
        self.save_path = config['paths']['saved_model'].format(self.model_name)

        self.model = None
        self.optimal_cutoff = self.parameters['optimal_cutoff']
        self.is_fitted = False

    def load(self, save_path=None):
        save_path = self.save_path if save_path is None else save_path
        print('Loading saved {} model from {}...'.format(self.model_name, save_path))
        self.model = load_model(save_path, custom_objects=self.custom_objects)
        self.optimal_cutoff = self.parameters['optimal_cutoff']

    def train(self, x_train, y_train, x_valid=None, y_valid=None):
        print('Fitting {} (stage 1)...'.format(self.model_name))
        valid = [x_valid, y_valid] if x_valid is not None and y_valid is not None else None
        x_train = np.concatenate((x_train, np.asarray([np.fliplr(image) for image in x_train])))
        y_train = np.concatenate((y_train, np.asarray([np.fliplr(image) for image in y_train])))

        if self.parameters['optimizer_stage1'] == 'adam':
            optim1 = adam(lr=self.parameters['lr_stage1'])
        elif self.parameters['optimizer_stage1'] == 'sgd':
            optim1 = sgd(lr=self.parameters['lr_stage1'], momentum=self.parameters['optimizer_momentum_stage1'])
        else:
            raise NotImplementedError('Optimizer should be either adam or SGD.')
        self.model.compile(optimizer=optim1, loss='binary_crossentropy', metrics=[iou])

        callbacks = [
            EarlyStopping(monitor='iou', patience=self.parameters['early_stopping'], verbose=2, mode='max'),
            ReduceLROnPlateau(monitor='iou', factor=0.5, patience=5, verbose=2, mode='max', min_lr=0.0001),
            ModelCheckpoint(self.save_path, monitor='iou', mode='max', verbose=2, save_best_only=True)
        ]

        self.model.fit(x=x_train, y=y_train, batch_size=self.parameters['batch_size_stage1'], shuffle=False,
                       epochs=self.parameters['epochs_stage1'], verbose=2, callbacks=callbacks, validation_data=valid)

        print('Fitting {} (stage 2)...'.format(self.model_name))
        self.model = Model(self.model.layers[0].input, self.model.layers[-1].input)

        if self.parameters['optimizer_stage2'] == 'adam':
            optim2 = adam(lr=self.parameters['lr_stage2'])
        elif self.parameters['optimizer_stage2'] == 'sgd':
            optim2 = sgd(lr=self.parameters['lr_stage2'], momentum=self.parameters['optimizer_momentum_stage2'])
        else:
            raise NotImplementedError('Optimizer should be either adam or SGD.')
        self.model.compile(optimizer=optim2, loss=lovasz_loss, metrics=[iou_no_sigmoid])

        for cb in callbacks:
            cb.monitor = 'iou_no_sigmoid'

        self.model.fit(x=x_train, y=y_train, batch_size=self.parameters['batch_size_stage2'], shuffle=False,
                       epochs=self.parameters['epochs_stage2'], verbose=2, callbacks=callbacks, validation_data=valid)
        self.load(self.save_path)

        # Determine the optimal likelihood cutoff for segmenting images
        if valid is not None:
            valid_predictions = self.model.predict(x_valid)
            self.optimal_cutoff = get_optimal_cutoff(valid_predictions, y_valid)

    def predict(self, x):
        print('Making predictions with {}...'.format(self.model_name))

        x_mirrored = np.array([np.fliplr(image) for image in x])
        predictions = self.model.predict(x).reshape(-1, 101, 101)
        predictions_mirrored = self.model.predict(x_mirrored).reshape(-1, 101, 101)
        predictions += np.array([np.fliplr(image) for image in predictions_mirrored]) / 2
        return predictions

    @staticmethod
    def preprocess(x):
        if isinstance(x, pd.Series):
            x = x.tolist()
        return np.asarray(x).reshape(-1, 101, 101, 1)

    @staticmethod
    def postprocess(x):
        return x

    def build(self):
        input_layer = Input(shape=(self.image_width, self.image_width, 1))

        # Get the needed parameters
        n_init = self.parameters['neurons_initial']
        ki = self.parameters['kernel_initializer']
        use_expansion_dropout = self.parameters['dropout_expansion']
        use_contraction_dropout = self.parameters['dropout_contraction']
        do_ratios_exp = self.parameters['dropout_ratios_expansion']
        do_ratios_con = self.parameters['dropout_ratios_contraction']
        bn_momentum = self.parameters['batchnorm_momentum']

        # 101 -> 50
        c1 = Conv2D(n_init, (3, 3), padding='same', kernel_initializer=ki)(input_layer)
        c1 = CustomResNet.residual_block(c1, n_init, False, bn_momentum)
        c1 = CustomResNet.residual_block(c1, n_init, True, bn_momentum)
        p1 = MaxPooling2D(pool_size=(2, 2))(c1)
        if use_expansion_dropout:
            p1 = Dropout(rate=do_ratios_exp[0], seed=1)(p1)

        # 50 -> 25
        c2 = Conv2D(n_init * 2, (3, 3), padding='same', kernel_initializer=ki)(p1)
        c2 = CustomResNet.residual_block(c2, n_init * 2, False, bn_momentum)
        c2 = CustomResNet.residual_block(c2, n_init * 2, True, bn_momentum)
        p2 = MaxPooling2D(pool_size=(2, 2))(c2)
        if use_expansion_dropout:
            p2 = Dropout(rate=do_ratios_exp[1])(p2)

        # 25 -> 12
        c3 = Conv2D(n_init * 4, (3, 3), padding='same', kernel_initializer=ki)(p2)
        c3 = CustomResNet.residual_block(c3, n_init * 4, False, bn_momentum)
        c3 = CustomResNet.residual_block(c3, n_init * 4, True, bn_momentum)
        p3 = MaxPooling2D(pool_size=(2, 2))(c3)
        if use_expansion_dropout:
            p3 = Dropout(rate=do_ratios_exp[2])(p3)

        # 12 -> 6
        c4 = Conv2D(n_init * 8, (3, 3), padding='same', kernel_initializer=ki)(p3)
        c4 = CustomResNet.residual_block(c4, n_init * 8, False, bn_momentum)
        c4 = CustomResNet.residual_block(c4, n_init * 8, True, bn_momentum)
        p4 = MaxPooling2D(pool_size=(2, 2))(c4)
        if use_expansion_dropout:
            p4 = Dropout(rate=do_ratios_exp[3])(p4)

        # Middle
        cm = Conv2D(n_init * 16, (3, 3), padding='same', kernel_initializer=ki)(p4)
        cm = CustomResNet.residual_block(cm, n_init * 16, False, bn_momentum)
        cm = CustomResNet.residual_block(cm, n_init * 16, True, bn_momentum)

        # 6 -> 12
        d4 = Conv2DTranspose(n_init * 8, (3, 3), strides=(2, 2), padding='same', kernel_initializer=ki)(cm)
        u4 = concatenate([d4, c4])
        if use_contraction_dropout:
            u4 = Dropout(rate=do_ratios_con[0])(u4)

        u4 = Conv2D(n_init * 8, (3, 3), padding='same', kernel_initializer=ki)(u4)
        u4 = CustomResNet.residual_block(u4, n_init * 8, False, bn_momentum)
        u4 = CustomResNet.residual_block(u4, n_init * 8, True, bn_momentum)

        # 12 -> 25
        d3 = Conv2DTranspose(n_init * 4, (3, 3), strides=(2, 2), padding='valid', kernel_initializer=ki)(u4)
        u3 = concatenate([d3, c3])
        if use_contraction_dropout:
            u3 = Dropout(rate=do_ratios_con[1])(u3)

        u3 = Conv2D(n_init * 4, (3, 3), padding='same', kernel_initializer=ki)(u3)
        u3 = CustomResNet.residual_block(u3, n_init * 4, False, bn_momentum)
        u3 = CustomResNet.residual_block(u3, n_init * 4, True, bn_momentum)

        # 25 -> 50
        d2 = Conv2DTranspose(n_init * 2, (3, 3), strides=(2, 2), padding='same', kernel_initializer=ki)(u3)
        u2 = concatenate([d2, c2])
        if use_contraction_dropout:
            u2 = Dropout(rate=do_ratios_con[2])(u2)

        u2 = Conv2D(n_init * 2, (3, 3), padding='same', kernel_initializer=ki)(u2)
        u2 = CustomResNet.residual_block(u2, n_init * 2, False, bn_momentum)
        u2 = CustomResNet.residual_block(u2, n_init * 2, True, bn_momentum)

        # 50 -> 101
        d1 = Conv2DTranspose(n_init, (3, 3), strides=(2, 2), padding='valid', kernel_initializer=ki)(u2)
        u1 = concatenate([d1, c1])
        if use_contraction_dropout:
            u1 = Dropout(rate=do_ratios_con[3])(u1)

        u1 = Conv2D(n_init, (3, 3), padding='same', kernel_initializer=ki)(u1)
        u1 = CustomResNet.residual_block(u1, n_init, False, bn_momentum)
        u1 = CustomResNet.residual_block(u1, n_init, True, bn_momentum)

        # Output layer
        output_layer = Conv2D(1, (1, 1), padding='same', kernel_initializer=ki)(u1)
        output_layer = Activation('sigmoid')(output_layer)

        self.model = Model(input_layer, output_layer)

    @staticmethod
    def batch_activation(input_layer, bn_momentum):
        layer = BatchNormalization(momentum=bn_momentum)(input_layer)
        layer = Activation('relu')(layer)
        return layer

    @staticmethod
    def convolution_block(block_input, neurons, activation=True, kernel_init='glorot_normal', bn_momentum=0.99):
        conv = Conv2D(neurons, kernel_size=(3, 3), strides=(1, 1), padding='same',
                      kernel_initializer=kernel_init)(block_input)
        if activation:
            conv = CustomResNet.batch_activation(conv, bn_momentum)
        return conv

    @staticmethod
    def residual_block(block_input, neurons, activation=False, bn_momentum=0.99):
        layer = CustomResNet.batch_activation(block_input, bn_momentum)
        layer = CustomResNet.convolution_block(layer, neurons)
        layer = CustomResNet.convolution_block(layer, neurons, activation=False)
        layer = Add()([layer, block_input])
        if activation:
            layer = CustomResNet.batch_activation(layer, bn_momentum)
        return layer
