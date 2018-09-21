"""
This file creates predictions for the "TGS Salt Identification Challenge" on
Kaggle. Information about the competition can be found here:

https://www.kaggle.com/c/tgs-salt-identification-challenge

The approach used here is an ensemble of convolutional neural networks. For a
more in-depth description of the project and approach, see `README.md`.

Author: Sam Waterbury
GitHub: https://github.com/samwaterbury/salt-identification
"""

import os
import sys
import json
import pickle
import pandas as pd
import numpy as np

from keras.preprocessing.image import load_img

from utilities import Logger, encode
from models.CustomResNet import CustomResNet
from models.ResNet50 import ResNet50


def main():
    """Runs the entire modeling process and generates predictions."""

    # Load the config file
    os.chdir(os.path.dirname(__file__))
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config.json'
    with open(config_path, mode='r') as config:
        parameters = json.load(config)

    # Write to terminal and log file simultaneously.
    sys.stdout = Logger(sys.stdout, log_path=parameters['filepaths']['logfile'])

    # Create missing directories if necessary
    for directory in parameters['mandatory_directories']:
        if not os.path.exists(directory):
            os.mkdir(directory)
    for directory in parameters['mandatory_directories']:
        if not os.listdir(directory):
            raise FileNotFoundError('Need to download the data! Check the readme.')

    # Construct the data set
    train, test = construct_data(parameters['filepaths'])

    if 'unet_resnet' in parameters['models_to_include']:
        # Model 1: U-Net with CustomResNet blocks
        print('Model 1: U-Net with CustomResNet blocks')
        model1 = UNetResNet(parameters['model_parameters']['unet_resnet'])
        model1.train(train)
        print('Making predictions...')
        predictions, optimal_cutoff1 = model1.predict(test['image'])
        predictions = np.round(predictions > optimal_cutoff1)

    if 'resnet50' in parameters['models_to_include']:
        # Model 2: ResNet34
        print('Model 2: ResNet34 with pretrained weights')
        model2 = ResNet50(parameters['model_parameters']['resnet50'])
        model2.train(train)
        print('Making predictions...')
        predictions, optimal_cutoff2 = model2.predict(test['image'])
        predictions = np.round(predictions > optimal_cutoff2)

    assert predictions

    # Encode predictions and write to submission file
    print('Encoding and saving the predictions...')
    predictions = pd.DataFrame.from_dict({
        i: encode(predictions[j]) for j, i in enumerate(test.index.values)
    }, orient='index')
    predictions.index.names = ['id']
    predictions.columns = ['rle_mask']
    predictions.to_csv(parameters['filepaths']['submission'])


def construct_data(filepaths):
    """
    Constructs the standard dataset to be used by all models.

    :return: DataFrames `train` and `test` with image, mask, and depth columns.
    """
    # If possible, read the constructed data from existing files
    if os.path.exists(filepaths['saved_train']) \
            and os.path.exists(filepaths['saved_test']):
        print('Found existing saved dataset; loading it...')
        with open(filepaths['saved_train'], mode='rb') as train_file:
            df_train = pickle.load(train_file)
        with open(filepaths['saved_test'], mode='rb') as test_file:
            df_test = pickle.load(test_file)
        return df_train, df_test

    print('Constructing the dataset...')

    # Read in the CSV files and create DataFrames for train, test observations
    depths = pd.read_csv(filepaths['depths_df'], index_col='id')
    df_train = pd.read_csv(filepaths['train_df'], index_col='id', usecols=[0]).join(depths)
    df_test = depths[~depths.index.isin(df_train.index)].copy()

    # Read in the images as greyscale and normalize the pixel values
    # (Training images)
    df_train['image'] = [np.array(load_img(filepaths['train_image'].format(i), color_mode='grayscale')) / 255
                         for i in df_train.index]
    # (Training masks)
    df_train['mask'] = [np.array(load_img(filepaths['train_mask'].format(i), color_mode='grayscale')) / 255
                        for i in df_train.index]
    # (Testing images)
    df_test['image'] = [np.array(load_img(filepaths['test_image'].format(i), color_mode='grayscale')) / 255
                        for i in df_test.index]

    # Calculate the coverage for the training images
    # Then, bin the images into discrete classes corresponding to their coverage
    df_train['coverage'] = df_train['mask'].map(np.sum) / pow(101, 2)
    df_train['coverage_class'] = df_train['coverage'].map(lambda cov: np.int(np.ceil(cov * 10)))

    # Write to file
    try:
        with open(filepaths['saved_train'], mode='wb') as train_file:
            pickle.dump(df_train, train_file)
        with open(filepaths['saved_test'], mode='wb') as test_file:
            pickle.dump(df_test, test_file)
    except OSError:
        print('Could not save the data due to an occasional Python bug in macOS.')

    return df_train, df_test


if __name__ == '__main__':
    main()
