"""
Procedural version of the entire program for testing purposes.
"""

import os
import sys
import json
import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from keras.preprocessing.image import load_img

from utilities import Logger, encode
from models.CustomResNet import CustomResNet
from models.ResNet50 import ResNet50

"""
Main function.
"""

# Load the config file
os.chdir(os.path.dirname(__file__))
config_path = sys.argv[1] if len(sys.argv) > 1 else 'config.json'
with open(config_path, mode='r') as config:
    parameters = json.load(config)

"""
Construct the data set.
"""
print('Constructing the dataset...')

filepaths = parameters['filepaths']

# If possible, read the constructed data from existing files
if os.path.exists(filepaths['saved_train']) and os.path.exists(filepaths['saved_test']):
    print('Found existing saved dataset; loading it...')
    with open(filepaths['saved_train'], mode='rb') as train_file:
        df_train = pickle.load(train_file)
    with open(filepaths['saved_test'], mode='rb') as test_file:
        df_test = pickle.load(test_file)
else:
    for directory in parameters['mandatory_directories']:
        if not os.path.exists(directory):
            os.mkdir(directory)
    for directory in parameters['mandatory_directories']:
        if not os.listdir(directory):
            raise FileNotFoundError('Need to download the data! Check the readme.')

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

"""
Create and execute the models.
"""

train = df_train
test = df_test

# SET SOME PARAMETERS
parameters['models_to_include'] = ['CustomResNet', 'ResNet50']

if 'CustomResNet' in parameters['models_to_run']:
    model_parameters = parameters['model_parameters']['CustomResNet']
    model = CustomResNet(model_parameters)

    if model_parameters['use_saved_model']:
        model.load(model_parameters['save_path'])
    elif parameters['final_predictions']:
        x_train, y_train = model.preprocess(train['image'], train['mask'])
        model.train(x_train, y_train, update_cutoff=False)
    else:
        x_train, y_train, x_valid, y_valid = train_test_split(*model.preprocess(train['image'], train['mask']),
                                                              random_state=1, stratify=train['coverage_class'])
        model.train(x_train, y_train, x_valid, y_valid)

    predictions, optimal_cutoff = model.predict(test['image'])
    predictions = np.round(predictions > optimal_cutoff)
    predictions = model.postprocess(predictions)

assert predictions

# Encode predictions and write to submission file
# print('Encoding and saving the predictions...')
# predictions = pd.DataFrame.from_dict({
#     i: encode(predictions[j]) for j, i in enumerate(test.index.values)
# }, orient='index')
# predictions.index.names = ['id']
# predictions.columns = ['rle_mask']
# predictions.to_csv(parameters['filepaths']['submission'])
