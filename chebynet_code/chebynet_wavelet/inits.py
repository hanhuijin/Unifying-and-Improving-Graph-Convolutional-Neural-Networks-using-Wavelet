# noinspection PyUnresolvedReferences
import tensorflow.compat.v1 as tf
import numpy as np


# 产生一个维度为shape的Tensor，值分布在（-1.0-1.0）之间，且为均匀分布
def uniform(shape, scale=1.0, name=None):
    """Uniform init."""
    initial = tf.random_uniform(shape, minval=0.0, maxval=scale, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def ones(shape, name=None):
    """All ones."""
    initial = tf.ones(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)
