"""
This file implements several useful functions for the modeling process.

Author: Sam Waterbury
GitHub: https://github.com/samwaterbury/salt-identification
"""

import numpy as np
from PIL import Image
from skimage.transform import resize
import tensorflow as tf
from keras import backend, losses

# ---------------------------- IMAGE MANIPULATION ---------------------------- #


def upsample(image, shape=(128, 128, 1)):
    return resize(image, output_shape=shape, mode='constant',
                  preserve_range=True, anti_aliasing=None)


def downsample(image, shape=(101, 101, 1)):
    return np.squeeze(resize(image, output_shape=shape, mode='constant',
                             preserve_range=True, anti_aliasing=None))


def read_image(path):
    image = Image.open(path).convert('L')
    image = np.asarray(image, dtype='int32') / 255
    return upsample(image)


def encode_mask(mask):
    pixels = mask.reshape(mask.shape[0] * mask.shape[1], order='F')
    encoding = []
    start = None
    right = 0
    for i in range(len(pixels)):
        left = right
        right = pixels[i]

        if (left == 0) & (right == 1):
            start = i + 1
            encoding.append(start)
        if (left == 1) & (right == 0):
            encoding.append(i - start + 1)

    if right == 1:
        encoding.append(i - start + 2)

    def to_str(a):
        s = str(a[0])
        for i in range(1, len(a)):
            s = s + ' ' + str(a[i])
        return s

    if len(encoding) == 0:
        return None
    else:
        return to_str(encoding)


# ---------------------------- EVALUATION METRICS ---------------------------- #


def as_float(x):
    return backend.cast(x, backend.floatx())


def as_bool(x):
    return backend.cast(x, bool)


def iou_loss_core(y_true, y_pred):
    intersection = y_true * y_pred
    complement = 1 - y_true
    union = y_true + (complement * y_pred)

    numerator = backend.sum(intersection, axis=-1) + backend.epsilon()
    denominator = backend.sum(union, axis=-1) + backend.epsilon()
    return numerator / denominator


def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        backend.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return backend.mean(backend.stack(prec), axis=0)


def iou_bce_loss(y_true, y_pred):
    return losses.binary_crossentropy(y_true, y_pred) + 3 * (1 - iou_loss_core(y_true, y_pred))


def competition_metric(true, pred):
    thresholds = [0.5 + (i * .05) for i in range(10)]

    true = backend.batch_flatten(true)
    pred = backend.batch_flatten(pred)
    pred = as_float(backend.greater(pred, 0.5))

    true_sum = backend.sum(true, axis=-1)
    pred_sum = backend.sum(pred, axis=-1)

    true1 = as_float(backend.greater(true_sum, 1))
    pred1 = as_float(backend.greater(pred_sum, 1))

    true_positive_mask = as_bool(true1 * pred1)

    test_true = tf.boolean_mask(true, true_positive_mask)
    test_pred = tf.boolean_mask(pred, true_positive_mask)

    iou = iou_loss_core(test_true, test_pred)
    true_positives = [as_float(backend.greater(iou, tres)) for tres in thresholds]

    true_positives = backend.mean(backend.stack(true_positives, axis=-1), axis=-1)
    true_positives = backend.sum(true_positives)

    true_negatives = (1-true1) * (1 - pred1)
    true_negatives = backend.sum(true_negatives)

    return (true_positives + true_negatives) / as_float(backend.shape(true)[0])
