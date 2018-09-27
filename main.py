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
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split
from keras.preprocessing.image import load_img

from utilities import Logger, encode
from models.CustomResNet import CustomResNet
from models.ResNet34 import ResNet34
from models.ResNet50 import ResNet50


def main():
    """
    Runs all of the steps in sequence and saves the final predictions to a file.
    """

    # Get the config file for this run
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config.json'
    with open(config_path, mode='r') as config_file:
        config = json.load(config_file)

    # Write to the terminal and log file simultaneously
    sys.stdout = Logger(sys.stdout, log_path=config['paths']['logfile'])

    # Construct the dataset after verifying we have what we need
    if not verify_paths(config):
        print('Some of the required directories or data files could not be found.\n'
              'Before running this file, run `setup.sh` to create/download them.')
        sys.exit(1)
    train, test = construct_data(config)

    # Modeling
    models = []
    for model_name in config['models_to_run']:
        models.append(get_model(model_name, config, train))

    # Predictions
    x_test = test['image']
    predictions = []
    for model in models:
        print('Making predictions with {}...'.format(model.model_name))
        y = model.predict(model.preprocess(x_test))
        y = np.round(y > model.optimal_cutoff)
        y = model.postprocess(y)
        predictions.append(y)
        print('Optimal cutoff for {} is {}.'.format(model.model_name, model.optimal_cutoff))
    y_final = predictions[0]  # np.mean(predictions, axis=0)

    print('Encoding and saving final predictions...')
    final = pd.DataFrame.from_dict({
        index: encode(y_final[row_number]) for row_number, index in enumerate(test.index.values)
    }, orient='index')
    final.index.names = ['id']
    final.columns = ['rle_mask']
    final.to_csv(config['paths']['submission'].format(datetime.now().strftime('%Y%m%d_%I%M%p')))

    print('Done!')
    sys.exit(0)


def verify_paths(config):
    """
    Verifies that the necessary files and subdirectories are present.

    :param config: Config file dictionary.
    :return: True if all paths and files exist, False otherwise.
    """
    must_exist = [
        'dir_output',
        'dir_data',
        'dir_train_images',
        'dir_train_masks',
        'dir_test_images',
        'df_depths',
        'df_train'
    ]
    must_not_be_empty = [
        'dir_train_images',
        'dir_train_masks',
        'dir_test_images'
    ]

    paths = config['paths']
    for item in must_exist:
        if not os.path.exists(paths[item]):
            return False
    for item in must_not_be_empty:
        if not os.listdir(paths[item]):
            return False
    return True


def construct_data(config):
    """
    Loads or recreates the entire dataset and returns the test and train sets.

    :param config: Config file dictionary.
    :return: `train` and `test` DataFrames containing images, masks, etc.
    """
    paths = config['paths']

    # If possible, read the constructed data from existing files
    if os.path.exists(paths['saved_train']) and os.path.exists(paths['saved_test']):
        print('Found existing saved dataset; loading it...')
        with open(paths['saved_train'], mode='rb') as train_file:
            df_train = pickle.load(train_file)
        with open(paths['saved_test'], mode='rb') as test_file:
            df_test = pickle.load(test_file)
        return df_train, df_test

    # Read in the CSV files and create DataFrames for train, test observations
    depths = pd.read_csv(paths['df_depths'], index_col='id')
    df_train = pd.read_csv(paths['df_train'], index_col='id', usecols=[0]).join(depths)
    df_test = depths[~depths.index.isin(df_train.index)].copy()

    # Read the images as greyscale and normalize the pixel values to be in [0,1]
    # (Training images)
    print('Reading training images...')
    df_train['image'] = [np.array(load_img(paths['train_image'].format(i), color_mode='grayscale'))
                         / 255 for i in df_train.index]
    # (Training masks)
    print('Reading training masks...')
    df_train['mask'] = [np.array(load_img(paths['train_mask'].format(i), color_mode='grayscale'))
                        / 255 for i in df_train.index]
    # (Testing images)
    print('Reading test images...')
    df_test['image'] = [np.array(load_img(paths['test_image'].format(i), color_mode='grayscale'))
                        / 255 for i in df_test.index]

    # Calculate the coverage for the training images
    # Then, bin the images into discrete classes corresponding to their coverage
    df_train['coverage'] = df_train['mask'].map(np.sum) / pow(101, 2)
    df_train['coverage_class'] = df_train['coverage'].map(lambda cov: np.int(np.ceil(cov * 10)))

    # Write to file
    print('Saving the constructed dataset...')
    try:
        with open(paths['saved_train'], mode='wb') as train_file:
            pickle.dump(df_train, train_file)
        with open(paths['saved_test'], mode='wb') as test_file:
            pickle.dump(df_test, test_file)
    except OSError:
        print('Could not save the data due to an occasional Python bug in macOS. :(')

    return df_train, df_test


def get_model(model_name, config, train):
    """
    Constructs, loads/trains, and then returns an instance of a model.

    :param model_name: Name of the model's class.
    :param config: Config file dictionary.
    :param train: Training DataFrame generated by `construct_dataset()`.
    :return: A fitted instance of the model's class, ready to make predictions.
    """
    model_parameters = config['model_parameters'][model_name]
    model = eval(model_name)(config)

    save_path = config['paths']['saved_model'].format(model.model_name)
    if model_parameters['use_saved_model'] and os.path.exists(save_path):
        print('Loading saved {} model from {}...'.format(model.model_name, save_path))
        model.load(save_path)
    elif config['final_predictions']:
        model.build()
        model.train(train['image'], train['mask'])
    else:
        model.build()
        split = train_test_split(train['image'], train['mask'], random_state=1,
                                 test_size=(1 / model_parameters['k_folds']),
                                 stratify=train['coverage_class'])
        model.train(*split)
    return model


if __name__ == '__main__':
    main()
