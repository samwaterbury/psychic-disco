"""Contains support functions, classes, and constants used by the project.

Author: Sam Waterbury
GitHub: https://github.com/samwaterbury/salt-identification
"""

import sys
from os import path

import numpy as np
import tensorflow as tf
from keras import backend

PROJECT_ROOT = path.abspath(path.join(path.dirname(__file__), path.pardir))
DEFAULT_PATHS = {
    'dir_data': path.join(PROJECT_ROOT, 'data/'),
    'dir_output': path.join(PROJECT_ROOT, 'output/'),
    'dir_train_images': path.join(PROJECT_ROOT, 'data/train/images/'),
    'dir_train_masks': path.join(PROJECT_ROOT, 'data/train/masks/'),
    'dir_test_images': path.join(PROJECT_ROOT, 'data/test/images/'),
    'df_depths': path.join(PROJECT_ROOT, 'data/depths.csv'),
    'df_train': path.join(PROJECT_ROOT, 'data/train.csv'),
}


class Logger(object):
    """Writes to the terminal and a logfile simultaneously.

    By replacing `sys.stdout` with an instance of this class, all output will
    be printed to both the console and the specified log file. This class will
    automatically save the original `sys.stdout` as an attribute.

    Attributes:
        stdout: The original `sys.stdout` object.

    Args:
        logfile: Filepath to be used or created for the logfile.
    """
    def __init__(self, logfile):
        self.stdout = sys.stdout
        self._log = open(logfile, mode='a')

    def write(self, message):
        self.stdout.write(message)
        self._log.write(message)

    def flush(self):
        pass

    def close(self):
        """Closes the logfile object and returns the original `sys.stdout`."""
        self._log.close()
        return self.stdout


def encode_predictions(image):
    """Encodes an image array in the proper string format for submission.

    Args:
        image: 2-dimensional mask array with pixel values in {0,1}.

    Returns:
        String containing formatted pixel coordinates.
    """
    pixels = np.concatenate([[0], image.flatten(order='F'), [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(i) for i in runs)


def get_cutoff(y_scores, y_true):
    """Finds the optimal cutoff point to use when classifying pixels.

    The model produces scores between -Inf and Inf for each pixel. To make
    predictions, the pixels need to be labeled 0 or 1 based on whether their
    scores are less than or greater than some cutoff value.

    Instead of using a cutoff of 0, this function tests many different potential
    cutoff points between ln(0.3) and ln(0.7) using validation data and a close
    approximator of the official competition scoring metric.

    Args:
        y_scores: (101, 101) arrays of pixel scores.
        y_true: (101, 101) arrays of the true masks.

    Returns:
        Approximate estimate of the optimal cutoff value.
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


def competition_metric(true, pred):
    """Official scoring metric for the competition.

    The competition accuracy score is given by the mean of

        Metric(t) = #TruePos(t) / [#TruePos(t) + #FalsePos(t) + #FalseNeg(t)]

    evaluated at thresholds t = {0.05, 0.10, ..., 0.90, 0.95}.

    Args:
        true: Numpy array containing the true mask.
        pred: Numpy array containing the predicted mask.

    Returns:
        Mean of Metric(t) evaluated at all thresholds.
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


def iou_sigmoid(y_true, y_scores):
    """TensorFlow wrapper for `competition_metric()`.

    Args:
        y_scores: 2-dimensional arrays of pixel scores.
        y_true: True masks corresponding to the arrays in `y_scores`.

    Returns:
        Evaluation of the competition metric for `y_scores`.
    """
    return tf.py_func(competition_metric, [y_true, y_scores > 0.5], tf.float64)


def iou_no_sigmoid(y_true, y_scores):
    """TensorFlow wrapper for `competition_metric()`.

    Args:
        y_scores: 2-dimensional arrays of pixel scores.
        y_true: True masks corresponding to the arrays in `y_scores`.

    Returns:
        Evaluation of the competition metric for `y_scores`.
    """
    return tf.py_func(competition_metric, [y_true, y_scores > 0.], tf.float64)


def lovasz_loss(y_true, y_pred):
    """Suitable proxy for the competition metric to use during model training.

    Args:
        y_true: True masks corresponding to the arrays in `y_scores`.
        y_pred: 2-dimensional arrays of pixel scores.

    Returns:
        Loss value.
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
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and
        +\infty)
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
      logits: [P] Variable, logits at each prediction (between -\infty and
        +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """

    def compute_loss():
        labelsf = tf.cast(labels, logits.dtype)
        signs = 2. * labelsf - 1.
        errors = 1. - logits * tf.stop_gradient(signs)
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0],
                                          name="descending_sort")
        gt_sorted = tf.gather(labelsf, perm)
        grad = lovasz_grad(gt_sorted)
        _loss = tf.tensordot(tf.nn.relu(errors_sorted), tf.stop_gradient(grad),
                             1, name="loss_non_void")
        return _loss

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
