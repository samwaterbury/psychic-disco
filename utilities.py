"""
This file implements several functions used throughout the project.

Author: Sam Waterbury
GitHub: https://github.com/samwaterbury/salt-identification
"""

import numpy as np
import tensorflow as tf
from skimage.transform import resize


# Constant paths to files in this repository
PATHS = {
    'train_images': 'data/train/images/',
    'test_images': 'data/test/images/',
    'train_masks': 'data/train/masks/',
    'depths_df': 'data/depths.csv',
    'train_df': 'data/train.csv',
    'saved_train': 'output/train.pk',
    'saved_test': 'output/test.pk',
    'saved_unetresnet': 'output/unetresnet.model',
    'saved_unet': 'output/unet.model',
    'submission': 'output/submission.csv'
}


def get_coverage_class(coverage):
    """
    Assigns a class 1,...,10 to a coverage score; used to stratify the training-
    validation split.

    :param coverage: Proportion of the pixels in an image which are salt-masked.
    :return: Intger between 1 and 10.
    """
    for cutoff in range(0, 11):
        if coverage * 10 <= cutoff:
            return cutoff


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


def get_iou(y_true, y_scores):
    """
    TensorFlow wrapper for `competition_metric()`.

    :param y_scores: 2-dimensional arrays of pixel scores.
    :param y_true: True masks corresponding to the arrays in `y_scores`.
    :return: Evaluation of the competition metric for `y_scores`.
    """
    return tf.py_func(competition_metric, [y_true, y_scores > 0.5], tf.float64)
