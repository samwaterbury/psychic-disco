"""
This file implements several functions used throughout the project.

Author: Sam Waterbury
GitHub: https://github.com/samwaterbury/salt-identification
"""

import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from skimage.transform import resize
from keras import backend


class Logger:
    """
    Writes all output to the terminal as well as a logfile.
    """
    def __init__(self, stdout, log_path):
        self.terminal = stdout
        if not os.path.exists(os.path.dirname(log_path)):
            os.mkdir(os.path.dirname(log_path))
        self.log = open(log_path.format(datetime.now().strftime('%Y%m%d-%I-%M-%p')), 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def upsample(image):
    """
    Resizes an image to a square 128x128 image by padding the edges.

    :param image: 101x101 array of pixel values.
    :return: 128x128 array of pixel values.
    """
    return resize(image, output_shape=(128, 128), mode='constant', preserve_range=True)


def downsample(image):
    """
    Resizes an image to a square 101x101 image by padding the edges.

    :param image: 128x128 array of pixel values.
    :return: 101x101 array of pixel values.
    """
    return resize(image, output_shape=(101, 101), mode='constant', preserve_range=True)


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


def iou(y_true, y_scores):
    """
    TensorFlow wrapper for `competition_metric()`.

    :param y_scores: 2-dimensional arrays of pixel scores.
    :param y_true: True masks corresponding to the arrays in `y_scores`.
    :return: Evaluation of the competition metric for `y_scores`.
    """
    return tf.py_func(competition_metric, [y_true, y_scores > 0.5], tf.float64)


def iou_no_sigmoid(y_true, y_scores):
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

# ---------------------------------------------------------------------------- #
# The following code is taken from this repository:
# https://github.com/bermanmaxim/LovaszSoftmax
#
# All credit goes to the original author:
# Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)
# ---------------------------------------------------------------------------- #


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
    return jaccard


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        def treat_image(log_lab):
            log, lab = log_lab
            log, lab = tf.expand_dims(log, 0), tf.expand_dims(lab, 0)
            log, lab = flatten_binary_scores(log, lab, ignore)
            return lovasz_hinge_flat(log, lab)
        losses = tf.map_fn(treat_image, (logits, labels), dtype=tf.float32)
        loss = tf.reduce_mean(losses)
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """

    def compute_loss():
        labelsf = tf.cast(labels, logits.dtype)
        signs = 2. * labelsf - 1.
        errors = 1. - logits * tf.stop_gradient(signs)
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort")
        gt_sorted = tf.gather(labelsf, perm)
        grad = lovasz_grad(gt_sorted)
        loss = tf.tensordot(tf.nn.relu(errors_sorted), tf.stop_gradient(grad), 1, name="loss_non_void")
        return loss

    # deal with the void prediction case (only void pixels)
    loss = tf.cond(tf.equal(tf.shape(logits)[0], 0),
                   lambda: tf.reduce_sum(logits) * 0.,
                   compute_loss,
                   strict=True,
                   name="loss"
                   )
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = tf.reshape(scores, (-1,))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return scores, labels
    valid = tf.not_equal(labels, ignore)
    vscores = tf.boolean_mask(scores, valid, name='valid_scores')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vscores, vlabels
