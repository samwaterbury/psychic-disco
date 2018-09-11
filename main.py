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
import pickle
import pandas as pd
import numpy as np

from keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split

from utilities import PATHS, get_coverage_class, get_optimal_cutoff, encode
from models import UNetResNet

fresh = False  # TODO Remove later


def construct_data(fresh=False):
    """
    Constructs the standard dataset to be used by all models.

    :return: DataFrames `train` and `test` with image, mask, and depth columns.
    """
    # Make sure the data has been downloaded and placed in the correct location
    for required_path in PATHS.values():
        if not os.path.exists(os.path.dirname(required_path)):
            if os.path.dirname(required_path).split('/') in ['train', 'test']:
                os.mkdir(os.path.dirname(required_path))
    if not os.listdir(PATHS['test_images']) or not os.listdir(PATHS['train_images']):
        raise FileNotFoundError('Need to download the data! Check the readme.')

    # If possible, read the constructed data from existing files
    if not fresh and os.path.exists(PATHS['saved_train']) and os.path.exists(PATHS['saved_test']):
        print('Found existing saved dataset; loading it...')
        with open(PATHS['saved_train'], mode='rb') as train_file:
            df_train = pickle.load(train_file)
        with open(PATHS['saved_test'], mode='rb') as test_file:
            df_test = pickle.load(test_file)
        return df_train, df_test

    print('Constructing the dataset...')

    # Read in the CSV files and create DataFrames for train, test observations
    depths = pd.read_csv(PATHS['depths_df'], index_col='id')
    df_train = pd.read_csv(PATHS['train_df'], index_col='id', usecols=[0]).join(depths)
    df_test = depths[~depths.index.isin(df_train.index)].copy()

    # Read in the images as greyscale and normalize the pixel values

    # (Training images)
    train_image_path = PATHS['train_images'] + '{}.png'
    df_train['image'] = [np.array(load_img(train_image_path.format(i), color_mode='grayscale')) / 255
                         for i in df_train.index]

    # (Training masks)
    train_mask_path = PATHS['train_masks'] + '{}.png'
    df_train['mask'] = [np.array(load_img(train_mask_path.format(i), color_mode='grayscale')) / 255
                        for i in df_train.index]

    # (Testing images)
    test_image_path = PATHS['test_images'] + '{}.png'
    df_test['image'] = [np.array(load_img(test_image_path.format(i), color_mode='grayscale')) / 255
                        for i in df_test.index]

    # Calculate the coverage for the training images
    # Then, bin the images into discrete classes corresponding to their coverage
    df_train['coverage'] = df_train['mask'].map(np.sum) / pow(101, 2)
    df_train['coverage_class'] = df_train['coverage'].map(get_coverage_class)

    # Write to file
    with open(PATHS['saved_train'], mode='wb') as train_file:
        pickle.dump(df_train, train_file)
    with open(PATHS['saved_test'], mode='wb') as test_file:
        pickle.dump(df_test, test_file)

    return df_train, df_test


# ---------------------------------------------------------------------------- #
# ---------------------------- DATA PREPROCESSING ---------------------------- #
# ---------------------------------------------------------------------------- #

# Construct the data set
train, test = construct_data(fresh)

# Split the st into a training set and a validation set
id_train, id_valid, x_train, x_valid, y_train, y_valid, cov_train, cov_test, depth_train, depth_test \
    = train_test_split(train.index.values,
                       np.array(train['image'].tolist()).reshape(-1, 101, 101, 1),
                       np.array(train['mask'].tolist()).reshape(-1, 101, 101, 1),
                       train['coverage'].values,
                       train['z'].values,
                       test_size=0.2, stratify=train['coverage_class'], random_state=1)

# Data augmentation
x_train = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
y_train = np.append(y_train, [np.fliplr(x) for x in y_train], axis=0)

# ---------------------------------------------------------------------------- #
# --------------------------------- MODELING --------------------------------- #
# ---------------------------------------------------------------------------- #

# Model 1: U-Net with ResNet blocks
print('Creating and fitting the U-Net with ResNet blocks...')
if not fresh and os.path.exists(PATHS['saved_unetresnet']):
    model1 = UNetResNet().load_model(load_path=PATHS['saved_unetresnet'])
else:
    model1 = UNetResNet(save_path=PATHS['saved_unetresnet'])
model1.fit_model(x_train, y_train, x_valid, y_valid)

# Make predictions for the validation set
model1_valid_pred = model1.predict(x_valid)

# Determine the optimal likelihood cutoff for segmenting images
model1_optimal_cutoff = get_optimal_cutoff(model1_valid_pred, y_valid)

# Make predictions for the test set
x_test = np.array(test['image'].tolist()).reshape(-1, 101, 101, 1)
model1_test_pred = model1.predict(x_test)

model1_predictions = pd.DataFrame.from_dict({
    i: encode(np.round(model1_test_pred[j] > model1_optimal_cutoff)) for j, i in enumerate(test.index.values)
}, orient='index')
model1_predictions.index.names = ['id']
model1_predictions.columns = ['rle_mask']
model1_predictions.to_csv(PATHS['submission'])
