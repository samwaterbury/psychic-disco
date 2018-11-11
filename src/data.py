"""Contains the functions which create and load the dataset.

The function `construct_dataset` will create and return the entire dataset. The
rest of the functions in this file are used by it internally.

Author: Sam Waterbury
GitHub: https://github.com/samwaterbury/salt-identification
"""

import os
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm
from keras.preprocessing.image import load_img

from src.utilities import DEFAULT_PATHS


def construct_data(paths=DEFAULT_PATHS, use_saved=True):
    """Loads or creates the entire dataset of images, masks, and features.

    Args:
        paths: Dict of paths to get the data files from.
        use_saved: If False, recreate the data from scratch even if a saved copy
            of the constructed data already exists.

    Returns:
        `train` and `test` DataFrames containing the following columns:
            id: Unique identifier for each image/mask pair.
            z: Depth at which the image was taken.
            image: 101x101 array of image pixel values.
            mask: 101x101 array of mask pixel values (only in `train`).
            coverage: % of the image containing salt (only in `train`).
            cov_class: Value 1-10 corresponding to coverage (only in `train`).
    """
    if not verify_paths(paths):
        raise FileNotFoundError('Some of the required data files could not be '
                                'found. Before running the project, run '
                                '`setup.sh` to create/download them.')

    # Paths to save or load the constructed datasets from
    saved_train = os.path.join(paths['dir_output'], 'train.pk')
    saved_test = os.path.join(paths['dir_output'], 'test.pk')

    # Load the data if possible
    if use_saved and os.path.exists(saved_train) and os.path.exists(saved_test):
        print('Found existing saved dataset; loading it...')
        with open(saved_train, mode='rb') as train_file:
            train = pickle.load(train_file)
        with open(saved_test, mode='rb') as test_file:
            test = pickle.load(test_file)
        return train, test

    print('Constructing dataset...')

    # Read in the .csv files and create DataFrames for train, test observations
    depths = pd.read_csv(paths['df_depths'], index_col='id')
    train = pd.read_csv(paths['df_train'], index_col='id', usecols=[0])
    train = train.join(depths)
    test = depths[~depths.index.isin(train.index)].copy()

    # (Training images)
    print('Reading training images...')
    path = paths['dir_train_images'] + '{}.png'
    train['image'] = [read_image(path.format(img)) for img in tqdm(train.index)]

    # (Training masks)
    print('Reading training masks...')
    path = paths['dir_train_masks'] + '{}.png'
    train['mask'] = [read_image(path.format(img)) for img in tqdm(train.index)]

    # (Testing images)
    print('Reading test images...')
    path = paths['dir_test_images'] + '{}.png'
    test['image'] = [read_image(path.format(img)) for img in tqdm(test.index)]

    # Calculate the coverage for the training images
    # Then, bin the images into discrete classes corresponding to their coverage
    train['coverage'] = train['mask'].map(np.sum) / pow(101, 2)
    train['cov_class'] = train['coverage'].map(
        lambda cov: np.int(np.ceil(cov * 10)))

    # Write to file
    print('Saving the constructed dataset...')
    try:
        with open(saved_train, mode='wb') as train_file:
            pickle.dump(train, train_file)
        with open(saved_test, mode='wb') as test_file:
            pickle.dump(test, test_file)
    except OSError:
        print('Could not save the data due to an occasional Python bug on some '
              'systems. :( If this is happening on macOS, try running on Linux '
              'instead.')

    return train, test


def verify_paths(paths=DEFAULT_PATHS):
    """Checks if the files and directories needed for the project exist.

    Args:
        paths: List or dictionary of paths to check.

    Returns:
        True if all necessary paths exist or could be created, False otherwise.
    """
    if isinstance(paths, dict):
        paths = list(paths.values())
    for path in paths:
        if os.path.exists(path):
            continue
        if os.path.isdir(path):
            os.mkdir(path)
            continue
        return False
    return True


def read_image(image_path):
    """Reads an image file as a normalized grayscale numpy array.

    Args:
        image_path: Filepath of .png file.

    Returns:
        Numpy array with shape (101, 101, 1) containing pixel values in [0, 1].
    """
    return np.array(load_img(image_path, grayscale=True)) / 255
