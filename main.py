"""
This file creates predictions for the "TGS Salt Identification Challenge" on
Kaggle. Information about the competition can be found here:

https://www.kaggle.com/c/tgs-salt-identification-challenge

The approach used here is a convolutional neural network. For a more in-depth
description of the project and approach, see `README.md`.

Author: Sam Waterbury
GitHub: https://github.com/samwaterbury/salt-identification
"""

import os
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from model import fit_model
from utilities import read_image, downsample, encode_mask

# Define constant filepaths
IMG_PATH_TRAIN = 'data/train/images/'
IMG_PATH_TEST = 'data/test/images/'
MASKS_PATH_TRAIN = 'data/train/masks/'
TRAIN_FILEPATH = 'output/train.pk'
TEST_FILEPATH = 'output/test.pk'
NETWORK_FILEPATH = 'output/network.pk'
MODEL_WEIGHTS_FILEPATH = 'output/weights.model'


def main(fresh=False):
    """
    Run all steps in sequence.

    :param fresh: Ignore previous results and rerun every step from scratch.
    """
    pass


def construct_dataset(limit=None):
    """
    Construct two identically-formatted datasets for test and train containing:
        id          :   ID of the image and its corresponding mask
        image       :   array of the image
        mask        :   array of the mask or `np.nan` if test observation
        depth       :   depth the image was taken at
        class       :   one of 10 classes 0,...,10 assigned according to depth

    :return: Dicts train, test containing the ids, images, masks, and depths.
    """
    train = {'id': [], 'image': [], 'mask': [], 'depth': [], 'coverage': [], 'class': []}
    test = {'id': [], 'image': [], 'mask': [], 'depth': [], 'coverage': [], 'class': []}

    # This file contains the depth corresponding to every image in train or test
    depths = pd.read_csv('data/depths.csv', index_col='id')

    # Collect the train data
    i = 0
    for filename in os.listdir(IMG_PATH_TRAIN):
        if limit is not None:
            if i > limit:
                break
        image_id = filename.split('.')[0]
        mask = read_image(os.path.join(MASKS_PATH_TRAIN, filename))
        coverage = np.sum(mask) / 101 ** 2

        # Detemine which coverage class this observation belongs to
        coverage_class = 10.
        for cutoff in range(11):
            if coverage * 10 <= cutoff:
                coverage_class = cutoff
                break

        # Enter this observation into the dictionary
        train['id'].append(image_id)
        train['image'].append(read_image(os.path.join(IMG_PATH_TRAIN, filename)))
        train['mask'].append(mask)
        train['depth'].append(depths.loc[image_id, 'z'])
        train['coverage'].append(coverage)
        train['class'].append(coverage_class)

    # Now do the same for the test data
    i = 0
    for filename in os.listdir(IMG_PATH_TEST):
        if limit is not None:
            if i > limit:
                break
        image_id = filename.split('.')[0]

        # Enter this observation into the dictionary
        test['id'].append(image_id)
        test['image'].append(read_image(os.path.join(IMG_PATH_TEST, filename)))
        test['depth'].append(depths.loc[image_id, 'z'])

    return train, test


if __name__ == '__main__':
    main()

# ---------------------------------------------------------------------------- #

fresh = False

# Make sure the data has been downloaded and placed in the correct location
for required_path in [IMG_PATH_TRAIN, IMG_PATH_TEST, MASKS_PATH_TRAIN]:
    if not os.path.exists(required_path):
        raise FileNotFoundError('The data is missing! Check the readme.')

# Construct the dataset
if os.path.exists(TRAIN_FILEPATH) and os.path.exists(TEST_FILEPATH) and not fresh:
    print('Existing dataset found; loading it...')
    with open(TRAIN_FILEPATH, mode='rb') as train_file:
        train = pickle.load(train_file)
    with open(TEST_FILEPATH, mode='rb') as test_file:
        test = pickle.load(test_file)
else:
    print('Constructing the dataset...')
    train, test = construct_dataset()
    try:
        with open(file=TRAIN_FILEPATH, mode='wb') as train_file:
            pickle.dump(train, file=train_file)
        with open(file=TEST_FILEPATH, mode='wb') as test_file:
            pickle.dump(test, test_file)
    except OSError:
        print('Couldn\'t save the dataset to disk due to an occasional bug in pickle! :(')

# Construct the training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(np.stack(train['image']), np.stack(train['mask']))

input('Press ENTER to continue with model fitting')

# Construct and fit the network
print('Building and fitting the model...')
model = fit_model(X_train, y_train, X_valid, y_valid, weights_filepath=MODEL_WEIGHTS_FILEPATH)

# Evaluate the model on the validation set
valid_results = model.evaluate(X_valid, y_valid)

# Now make predictions for the test set, downsample, and create the masks
print('Making predictions')
predictions = model.predict(test['image'], verbose=1)
predictions = [downsample(array) for array in predictions]
test['mask'] = [np.rint(array).astype(np.uint8) for array in predictions]
encoded_masks = [encode_mask(mask) for mask in test['mask']]
