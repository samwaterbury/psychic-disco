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

from tqdm import tqdm
from sklearn.model_selection import train_test_split

from model import fit_model
from utilities import read_image, downsample, encode_mask

# Define constant filepaths
PATHS = {
    'train_images': 'data/train/images/',
    'test_images': 'data/test/images/',
    'train_masks': 'data/train/masks/',
    'saved_train': 'output/train.pk',
    'saved_test': 'output/test.pk',
    'saved_weights': 'output/weights.model',
    'submission': 'output/submission.csv'
}

# IMG_PATH_TRAIN = 'data/train/images/'
# IMG_PATH_TEST = 'data/test/images/'
# MASKS_PATH_TRAIN = 'data/train/masks/'
# TRAIN_FILEPATH = 'output/train.pk'
# TEST_FILEPATH = 'output/test.pk'
# NETWORK_FILEPATH = 'output/network.pk'
# MODEL_WEIGHTS_FILEPATH = 'output/weights.model'


def main(fresh=False):
    """
    Run all steps in sequence.

    :param fresh: Ignore previous results and rerun every step from scratch.
    """
    # Make sure the data has been downloaded and placed in the correct location
    for required_path in PATHS.values():
        if not os.path.exists(os.path.dirname(required_path)):
            if os.path.dirname(required_path).split('/') in ['train', 'test']:
                os.mkdir(os.path.dirname(required_path))
    if not os.listdir(PATHS['test_images']) or not os.listdir(PATHS['train_images']):
        raise FileNotFoundError('Need to download the data! Check the readme.')

    # Construct the dataset
    if os.path.exists(PATHS['saved_train']) and os.path.exists(PATHS['saved_test']) and not fresh:
        print('Existing dataset found; loading it...')
        with open(PATHS['saved_train'], mode='rb') as train_file:
            train = pickle.load(train_file)
        with open(PATHS['saved_test'], mode='rb') as test_file:
            test = pickle.load(test_file)
    else:
        print('Constructing the dataset...')
        train, test = construct_dataset()
        try:
            with open(file=PATHS['saved_train'], mode='wb') as train_file:
                pickle.dump(train, file=train_file)
            with open(file=PATHS['saved_test'], mode='wb') as test_file:
                pickle.dump(test, test_file)
        except OSError:
            print('Couldn\'t save the dataset to disk due to an occasional pickle bug! :(')

    # Construct the training and validation sets
    x_train, x_val, y_train, y_val, cov_train, cov_val, depth_train, depth_val \
        = train_test_split(np.stack(train['image']), np.stack(train['mask']),
                           train['coverage'], train['depth'],
                           stratify=train['class'], test_size=0.2, random_state=1)

    # Augment the training data by mirroring the training observations
    x_train = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
    y_train = np.append(y_train, [np.fliplr(x) for x in y_train], axis=0)

    # Construct and fit the network
    print('Building and fitting the model...')
    model = fit_model(x_train, y_train, x_val, y_val, weights_filepath=PATHS['saved_weights'])

    # Now make predictions for the test set, downsample, and create the masks
    print('Making predictions')

    # TODO
    # predictions = model.predict(np.stack(test['image']), verbose=1)
    # predictions = [downsample(array) for array in predictions]
    # test['mask'] = [np.rint(array).astype(np.uint8) for array in predictions]
    # encoded_masks = [encode_mask(mask) for mask in test['mask']]
    # final = pd.DataFrame({'id': test['id'], 'rle_mask': encoded_masks}, index='id')


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
    for filename in tqdm(os.listdir(PATHS['train_images'])):
        image_id = filename.split('.')[0]
        mask = read_image(os.path.join(PATHS['train_masks'], filename))
        coverage = np.sum(mask) / 101 ** 2

        # Detemine which coverage class this observation belongs to
        coverage_class = 10.
        for cutoff in range(11):
            if coverage * 10 <= cutoff:
                coverage_class = cutoff
                break

        # Enter this observation into the dictionary
        train['id'].append(image_id)
        train['image'].append(read_image(os.path.join(PATHS['train_images'], filename)))
        train['mask'].append(mask)
        train['depth'].append(depths.loc[image_id, 'z'])
        train['coverage'].append(coverage)
        train['class'].append(coverage_class)

    # Now do the same for the test data
    for filename in tqdm(os.listdir(PATHS['test_images'])):
        image_id = filename.split('.')[0]

        # Enter this observation into the dictionary
        test['id'].append(image_id)
        test['image'].append(read_image(os.path.join(PATHS['test_images'], filename)))
        test['depth'].append(depths.loc[image_id, 'z'])

    return train, test


if __name__ == '__main__':
    main()
