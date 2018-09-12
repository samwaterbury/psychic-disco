"""
This file implements several functions used throughout the project.

Author: Sam Waterbury
GitHub: https://github.com/samwaterbury/salt-identification
"""

from datetime import datetime

import numpy as np
import tensorflow as tf
from skimage.transform import resize
from keras import backend

from lovasz import lovasz_hinge


# Constant paths to files in this repository
PATHS = {
    'train_images':     'data/train/images/',
    'test_images':      'data/test/images/',
    'train_masks':      'data/train/masks/',
    'depths_df':        'data/depths.csv',
    'train_df':         'data/train.csv',
    'saved_train':      'output/train.pk',
    'saved_test':       'output/test.pk',
    'save_model1':      'output/model1.model',
    'save_model2':      'output/model2.model',
    'logfile':          'output/log-{}',
    'submission':       'output/submission.csv'
}


class Logger:
    def __init__(self, stdout):
        self.terminal = stdout
        self.log = open(PATHS['logfile'].format(datetime.now().strftime('%Y%m%d-%I-%M-%p')), 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def resize_image(image, size=101):
    """
    Resizes an image to a square image with width `size`. If necessary, pads the
    edges of the photo with a single constant value.

    :param image: 2-dimensional array of pixel values.
    :param size: Intger width of the resized image.
    :return: 2-dimensional array of pixel values with dimensions (size, size).
    """
    return resize(image, output_shape=(size, size), mode='constant', preserve_range=True)


def get_optimal_cutoff(y_scores, y_true):
    """
    Determines the cutoff above which pixels should be added to the mask which
    maximizes the expected competition score.

    :param y_scores: 2-dimensional arrays of pixel scores.
    :param y_true: True masks corresponding to the arrays in `y_scores`.
    :return: Approximate optimal cutoff between 0.3 and 0.7.
    """
    cutoffs = np.linspace(0.3, 0.7, 31)
    cutoffs = np.log(cutoffs / (1 - cutoffs))

    metric_evaluations = []
    for cutoff in cutoffs:
        y_pred = np.array(y_scores) > cutoff
        metric = [competition_metric(y_true[batch], y_pred[batch])
                  for batch in range(y_true.shape[0])]
        metric_evaluations.append(np.mean(metric))
    metric_evaluations = np.array(metric_evaluations)
    return cutoffs[np.argmax(metric_evaluations)]


def encode(image):
    """
    Encodes an image array in the proper string format for submission.

    :param image: 2-dimensional mask array with pixel values in {0,1}.
    :return: String containing formatted pixel coordinates.
    """
    pixels = np.concatenate([[0], image.flatten(order='F'), [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(i) for i in runs)


def competition_metric(true, pred):
    """
    The official scoring metric for this competition is given by the mean of

        Metric(t) = #TruePos(t) / [#TruePos(t) + #FalsePos(t) + #FalseNeg(t)]

    evaluated at thresholds t = {0.05, 0.10, ..., 0.90, 0.95}.

    :param true: Numpy array of true mask.
    :param pred: Numpy array of predicted mask.
    :return: Mean of Metric(t) evaluated at all thresholds.
    """
    batch_size = true.shape[0]
    metric = []
    for batch in range(batch_size):
        batch_true = true[batch] > 0
        batch_pred = pred[batch] > 0

        intersection = np.logical_and(batch_true, batch_pred)
        union = np.logical_or(batch_true, batch_pred)
        iou = (np.sum(intersection > 0) + 1e-10) / (np.sum(union > 0) + 1e-10)

        s = []
        for thresh in np.arange(0.5, 1, 0.05):
            s.append(iou > thresh)
        metric.append(np.mean(s))
    return np.mean(metric)


def get_iou_round1(y_true, y_scores):
    """
    TensorFlow wrapper for `competition_metric()`.

    :param y_scores: 2-dimensional arrays of pixel scores.
    :param y_true: True masks corresponding to the arrays in `y_scores`.
    :return: Evaluation of the competition metric for `y_scores`.
    """
    return tf.py_func(competition_metric, [y_true, y_scores > 0.5], tf.float64)


def get_iou_round2(y_true, y_scores):
    """
    TensorFlow wrapper for `competition_metric()`.

    :param y_scores: 2-dimensional arrays of pixel scores.
    :param y_true: True masks corresponding to the arrays in `y_scores`.
    :return: Evaluation of the competition metric for `y_scores`.
    """
    return tf.py_func(competition_metric, [y_true, y_scores > 0.], tf.float64)


def lovasz_loss(y_true, y_pred):
    """
    Lovasz loss is a suitable proxy for the competition metric to use during
    model training.

    :param y_true: True masks corresponding to the arrays in `y_scores`.
    :param y_pred: 2-dimensional arrays of pixel scores.
    :return:
    """
    y_true = backend.cast(backend.squeeze(y_true, -1), 'int32')
    y_pred = backend.cast(backend.squeeze(y_pred, -1), 'float32')
    return lovasz_hinge(y_pred, y_true)
