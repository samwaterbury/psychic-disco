"""This file creates the final predictions by running all parts of the project.

Author: Sam Waterbury
GitHub: https://github.com/samwaterbury/salt-identification
"""

from datetime import datetime
import os
import sys
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.utilities import DEFAULT_PATHS, Logger, encode_predictions
from src.data import construct_data
from src.model import CustomResNet

DEFAULT_CONFIG = {
    'paths': DEFAULT_PATHS,
    'model_parameters': {},
    'use_saved_data': True,
    'use_saved_models': True,
    'k_folds': 1,
    'holdout_percent': 0.2
}


def main():
    """Runs all of the steps and saves the final predictions to a file."""
    run_datetime = datetime.now().strftime('%Y%m%d_%I%M%p')

    # Get the configuration settings if supplied; otherwise, use the default
    config = DEFAULT_CONFIG
    if len(sys.argv) > 1:
        with open(sys.argv[1], mode='r') as config_file:
            config.update(json.load(config_file))

    # Write to the terminal and log file simultaneously
    output_dir = config['paths']['dir_output']
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    logfile = os.path.join(output_dir, 'log-{}.log'.format(run_datetime))
    sys.stdout = Logger(logfile)

    train, test = construct_data(config['paths'], config['use_saved_data'])

    # Generate K sets of training, validation data from `train`
    train_index_folds = []
    valid_index_folds = []
    if config['k_folds'] == 1:
        split = train_test_split(train.index.values,
                                 random_state=1,
                                 test_size=config['holdout_percent'],
                                 stratify=train['cov_class'])
        split = [[split[0], split[1]]]
    else:
        skf = StratifiedKFold(n_splits=config['k_folds'], random_state=1)
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

    # Final predictions are made on `x_test` and averaged in `y_test`
    x_test = CustomResNet.process(test['image'])
    y_test = np.zeros((len(x_test), 101, 101))
    found_cutoff_values = []

    # Load or train a model for each fold and make predictions
    model_parameters = config['model_parameters']
    for i, (train_indices, valid_indices) in enumerate(folds, start=1):
        model_parameters.update({'model_name': 'CustomResNet_{}'.format(i)})
        model = CustomResNet(**model_parameters)

        # Load the model weights or train the model
        if config['use_saved_models'] and os.path.exists(model.save_path):
            print('Found saved copy of model {}; loading it...'.format(i))
            model.load(model.save_path)
        else:
            print('Beginning model {} training...'.format(i))
            model.train(x_train=train.loc[train_indices, 'image'],
                        y_train=train.loc[train_indices, 'mask'],
                        x_valid=train.loc[valid_indices, 'image'],
                        y_valid=train.loc[valid_indices, 'mask'])

        y_test += model.predict(model.process(x_test))
        found_cutoff_values.append(model.optimal_cutoff)

    # Take the average and round pixels to 0 or 1 based on the cutoff value
    y_test /= config['k_folds']
    optimal_cutoff = np.mean(found_cutoff_values)
    y_test = np.round(y_test > optimal_cutoff)

    # Encode the predictions in the submission format
    print('Encoding and saving final predictions...')
    final = {}
    for row_number, idx in enumerate(test.index.values):
        final[idx] = encode_predictions(y_test[row_number])
    final = pd.DataFrame.from_dict(final, orient='index')
    final.index.names = ['id']
    final.columns = ['rle_mask']

    # Lastly, write the predictions to a .csv file
    final_path = os.path.join(
        output_dir, 'submission-{}.csv'.format(run_datetime))
    final.to_csv(final_path)
    print('Done! Predictions saved to {}.'.format(final_path))


if __name__ == '__main__':
    main()
