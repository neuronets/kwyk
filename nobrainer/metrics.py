# -*- coding: utf-8 -*-
"""Metrics implemented in TensorFlow and NumPy. Functions that end in
`_numpy` are implemented in NumPy for eager computation.

Functions that have the prefix `streaming_` are meant to be used when
evaluating TensorFlow estimators with `estimator.eval()`.
"""

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import add_to_collections

from nobrainer.util import _EPSILON

def dice_numpy(im1, im2, empty_score=1.0):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score
        
    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum

def dice(labels, predictions, axis=None, name=None):
    """Return the Dice coefficients between two Tensors. Perfect Dice is 1.0.

    The Dice coefficient between predictions `p` and labels `g` is

    .. math::
        \frac{\Sigma_i^N p_i g_i + \epsilon}
            {\Sigma_i^N p_i^2 + \Sigma_i^N g_i^2 + \epsilon}

            where `\epsilon` is a small value for stability.

    Parameters
    ----------
    labels: `Tensor`, input boolean tensor.
    predictions: `Tensor`, input boolean tensor.
    axis: `int` or `tuple`, the dimension(s) along which to compute Dice.
    name: `str`, a name for the operation.

    Returns
    -------
    `Tensor` of Dice coefficient(s).
    """
    with tf.variable_scope(name, 'dice', [predictions, labels]):
        labels = tf.convert_to_tensor(labels)
        predictions = tf.convert_to_tensor(predictions)
        predictions = tf.to_float(predictions)
        labels = tf.to_float(labels)
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())
        intersection = tf.reduce_sum(predictions * labels, axis=axis)
        union = (
            tf.reduce_sum(predictions, axis=axis)
            + tf.reduce_sum(labels, axis=axis))
        return (2 * intersection + _EPSILON) / (union + _EPSILON)


def streaming_dice(labels,
                   predictions,
                   axis,
                   weights=None,
                   metrics_collections=None,
                   update_collections=None,
                   name=None):
    """Calculates Dice coefficient between `labels` and `features`.

    Both tensors should have the same shape and should not be one-hot encoded.
    """
    values = dice(labels, predictions, axis=axis)
    mean_dice, update_op = tf.metrics.mean(values)

    if metrics_collections:
        add_to_collections(metrics_collections, mean_dice)
    if update_collections:
        add_to_collections(update_collections, update_op)

    return mean_dice, update_op


def hamming(labels, predictions, axis=None, name=None):
    """Return the Hamming distance between two Tensors.

    This operation cannot be used as a loss function because it uses `tf.cast`.

    Args:
        u, v: `Tensor`, input tensors.
        axis: `int` or `tuple`, the dimension(s) along which to compute Hamming
            distance.
        name: `str`, a name for the operation.
        dtype: `str` or dtype object, specified output type of the operation.
            Defaults to `tf.float64`.

    Returns:
        `Tensor` of Hamming distance(s).

    Notes:
        This function is almost identical to `scipy.spatial.distance.hamming`
        but accepts n-D tensors and adds an `axis` parameter.
    """
    with tf.name_scope(name, 'hamming', [labels, predictions]):
        labels = tf.convert_to_tensor(labels)
        predictions = tf.convert_to_tensor(predictions)
        ne = tf.not_equal(labels, predictions)
        return tf.reduce_mean(tf.to_float(ne), axis=axis)


def streaming_hamming(labels,
                      predictions,
                      axis=None,
                      weights=None,
                      metrics_collections=None,
                      update_collections=None,
                      name=None):
    """Calculates Hamming distance between `labels` and `features`.

    Axis should be `(1, 2, 3)` for 3D segmentation problem, where 0 is the
    batch dimension.
    """
    values = hamming(labels=labels, predictions=predictions, axis=axis)
    mean_value, update_op = tf.metrics.mean(values)

    if metrics_collections:
        add_to_collections(metrics_collections, mean_value)
    if update_collections:
        add_to_collections(update_collections, update_op)

    return mean_value, update_op
