"""
This file implements several useful functions for the modeling process.

Author: Sam Waterbury
GitHub: https://github.com/samwaterbury/salt-identification
"""

import numpy as np
from PIL import Image
from skimage.transform import resize
import tensorflow as tf

# ---------------------------- IMAGE MANIPULATION ---------------------------- #


def upsample(image, shape=(128, 128, 1)):
    return resize(image, output_shape=shape, mode='constant', cval=0,
                  preserve_range=True, anti_aliasing=None)


def downsample(image, shape=(101, 101, 1)):
    return np.squeeze(resize(image, output_shape=shape, mode='constant',
                             preserve_range=True, anti_aliasing=None))


def read_image(path):
    image = Image.open(path).convert('L')
    image = np.asarray(image, dtype='int32') / 255
    return upsample(image)


# ------------------------------- MODEL METRICS ------------------------------ #


def get_iou_vector(true, pred):
    """
    Computes the custom accuracy metric used in this competition. Credit goes to
    Alex Donchuk for this implementation:

    https://www.kaggle.com/donchuk/fast-implementation-of-scoring-metric

    :param true: True mask array.
    :param pred: Predicted mask array.
    :return: IOU score vector.
    """
    batch_size = true.shape[0]
    metric = []
    for batch in range(batch_size):
        t, p = true[batch] > 0, pred[batch] > 0

        intersection = np.logical_and(t, p)
        union = np.logical_or(t, p)
        iou = (np.sum(intersection > 0) + 1e-10) / (np.sum(union > 0) + 1e-10)
        thresholds = np.arange(0.5, 1, 0.05)
        s = []
        for thresh in thresholds:
            s.append(iou > thresh)
        metric.append(np.mean(s))

    return np.mean(metric)


def competition_metric(label, pred):
    return tf.py_func(get_iou_vector, [label, pred > 0.5], tf.float64)
