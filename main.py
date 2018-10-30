"""
This file creates predictions for the "TGS Salt Identification Challenge" on
Kaggle. Information about the competition can be found here:

https://www.kaggle.com/c/tgs-salt-identification-challenge

The approach used here is a convolutional neural network trained separately on
k folds and then averaged. For a more in-depth description of the project and
approach, see `README.md`.

Author: Sam Waterbury
GitHub: https://github.com/samwaterbury/salt-identification
"""

from datetime import datetime
import os
import sys
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from keras.preprocessing.image import load_img

from model import CustomResNet

DEFAULT_CONFIG = {
    'paths': {
        'dir_data': 'data/',
        'dir_output': 'output/',
        'dir_train_images': 'data/train/images/',
        'dir_train_masks': 'data/train/masks/',
        'dir_test_images': 'data/test/images/',
        'df_depths': 'data/depths.csv',
        'df_train': 'data/train.csv',
    }
}


class Logger(object):
    """Writes all output to the terminal and a logfile simultaneously."""
    def __init__(self, stdout, logfile):
        self.stdout = stdout
        self.log = open(logfile, mode='a')

    def write(self, message):
        self.stdout.write(message)
        self.log.write(message)

    def flush(self):
        pass


def main():
    """Runs all of the steps and saves the final predictions to a file."""
    run_datetime = datetime.now().strftime('%Y%m%d_%I%M%p')
    config = DEFAULT_CONFIG
    if len(sys.argv) > 1:
        with open(sys.argv[1], mode='r') as config_file:
            config.update(json.load(config_file))

    # Write to the terminal and log file simultaneously
    if not os.path.exists(config['paths']['dir_output']):
        os.mkdir(config['paths']['dir_output'])
    sys.stdout = Logger(sys.stdout, logfile=os.path.join(
        config['paths']['dir_output'], run_datetime))

    # Construct the dataset after verifying we have what we need
    if not verify_paths(config):
        print('Some of the required directories or data files could not be '
              'found.\nBefore running this file, run `setup.sh` to create/'
              'download them.')
        sys.exit(1)
    train, test = construct_data(config)

    # Modeling
    model_parameters = config.get('CustomResNet', {})
    k_folds = model_parameters.get('k_folds', 1)
    holdout = model_parameters.get('holdout_percent', 0.2)

    # Split the training data into K groups, stratifying by coverage
    train_index_folds = []
    valid_index_folds = []
    if k_folds == 1:
        split = train_test_split(train.index.values,
                                 random_state=1,
                                 test_size=holdout,
                                 stratify=train['cov_class'])
        split = [[split[0], split[1]]]
    else:
        skf = StratifiedKFold(n_splits=k_folds, random_state=1, shuffle=True)
        split = skf.split(train.index.values, train['cov_class'])
        _split = []
        for train_row_numbers, valid_row_numbers in split:
            _split.append([train.iloc[train_row_numbers].index,
                           train.iloc[valid_row_numbers].index])
        split = _split

    # Zip the train/valid indices so they can be iterated over in parallel
    for train_indices, valid_indices in split:
        train_index_folds.append(train_indices)
        valid_index_folds.append(valid_indices)
    folds = zip(train_index_folds, valid_index_folds)

    # Load or train the model(s)
    models = []
    for i, (train_indices, valid_indices) in enumerate(folds, start=1):
        print('Constructing model {}...'.format(i))

        model = CustomResNet(name='CustomResNet_{}'.format(i),
                             output=config['paths']['dir_output'],
                             **model_parameters)
        if not model.is_fitted:
            model.train(x_train=train.loc[train_indices, 'image'],
                        y_train=train.loc[train_indices, 'mask'],
                        x_valid=train.loc[valid_indices, 'image'],
                        y_valid=train.loc[valid_indices, 'mask'])
        models.append(model)

    # Make predictions with each model
    x_test = test['image']
    y_test = np.zeros((len(x_test), 101, 101))
    for i, model in enumerate(models, start=1):
        print('Making predictions with model {}...'.format(i))
        y_test += model.predict(model.process(x_test))

    # Take the average
    optimal_cutoff = np.mean([model.optimal_cutoff for model in models])
    print('Optimal cutoff is {}.'.format(optimal_cutoff))
    y_test /= k_folds
    y_test = np.round(y_test > optimal_cutoff)

    # Encode the predictions in the submission format
    print('Encoding and saving final predictions...')
    final = {}
    for row_number, idx in enumerate(test.index.values):
        final[idx] = encode(y_test[row_number])
    final = pd.DataFrame.from_dict(final, orient='index')
    final.index.names = ['id']
    final.columns = ['rle_mask']

    final_path = os.path.join(config['paths']['dir_output'],
                              'submission-{}.csv'.format(run_datetime))
    final.to_csv(final_path)
    print('Done!')


def verify_paths(config):
    """Verifies that the necessary files and subdirectories are present.

    Args:
        config: Dictionary with configuration settings.

    Returns:
        True if all paths are correct and files exist, False otherwise.
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
    """Loads or recreates the entire dataset.

    Args:
        config: Config file dictionary.

    Returns:
        `train` and `test` DataFrames containing the following columns:
            id: Unique identifier for each image/mask pair.
            z: Depth at which the image was taken.
            image: 101x101 array of image pixel values.
            mask: 101x101 array of mask pixel values (only in `train`).
            coverage: % of the image containing salt (only in `train`).
            cov_class: Value 1-10 corresponding to coverage (only in `train`).
    """
    paths = config['paths']
    save_train_path = os.path.join(config['paths']['dir_output'], 'train.pk')
    save_test_path = os.path.join(config['paths']['dir_output'], 'test.pk')

    # If possible, read the constructed data from existing files
    if os.path.exists(save_train_path) and os.path.exists(save_test_path):
        print('Found existing saved dataset; loading it...')
        with open(save_train_path, mode='rb') as train_file:
            df_train = pickle.load(train_file)
        with open(save_test_path, mode='rb') as test_file:
            df_test = pickle.load(test_file)
        return df_train, df_test

    # Read in the CSV files and create DataFrames for train, test observations
    depths = pd.read_csv(paths['df_depths'], index_col='id')
    df_train = pd.read_csv(paths['df_train'], index_col='id',
                           usecols=[0]).join(depths)
    df_test = depths[~depths.index.isin(df_train.index)].copy()

    # Read the images as greyscale and normalize the pixel values to be in [0,1]

    def get_image(image_path):
        """Loads an image as a grayscale numpy array."""
        return np.array(load_img(image_path, color_mode='grayscale')) / 255

    # (Training images)
    print('Reading training images...')
    path = paths['dir_train_images'] + '{}.png'
    df_train['image'] = [get_image(path.format(i)) for i in df_train.index]

    # (Training masks)
    print('Reading training masks...')
    path = paths['dir_train_masks'] + '{}.png'
    df_train['mask'] = [get_image(path.format(i)) for i in df_train.index]

    # (Testing images)
    print('Reading test images...')
    path = paths['dir_test_images'] + '{}.png'
    df_test['image'] = [get_image(path.format(i)) for i in df_test.index]

    # Calculate the coverage for the training images
    # Then, bin the images into discrete classes corresponding to their coverage
    df_train['coverage'] = df_train['mask'].map(np.sum) / pow(101, 2)
    df_train['cov_class'] = df_train['coverage'].map(
        lambda cov: np.int(np.ceil(cov * 10)))

    # Write to file
    print('Saving the constructed dataset...')
    try:
        with open(save_train_path, mode='wb') as train_file:
            pickle.dump(df_train, train_file)
        with open(save_test_path, mode='wb') as test_file:
            pickle.dump(df_test, test_file)
    except OSError:
        print('Could not save the data due to an occasional Python bug on some '
              'systems. :( If this is happening on macOS, try running on Linux '
              'instead.')

    return df_train, df_test


def encode(image):
    """Encodes an image array in the proper string format for submission.

    Args:
        image: 2-dimensional mask array with pixel values in {0,1}.

    Returns:
        String containing formatted pixel coordinates.
    """
    pixels = np.concatenate([[0], image.flatten(order='F'), [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(i) for i in runs)


if __name__ == '__main__':
    main()
