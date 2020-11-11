import numpy as np
import tensorflow as tf

def wrap_angle(angle):
    return ((angle - np.pi) % (2 * np.pi)) - np.pi

def et_distance(encoding_input,e_t):

    # tf.reduce_mean((encoding_input-e_t)**2,axis=-1)
    encoding_input=tf.nn.l2_normalize(encoding_input,epsilon=1e-12,axis=-1)
    e_t=tf.nn.l2_normalize(e_t,epsilon=1e-12,axis=-1)
    cosd=tf.ones([tf.shape(e_t)[0],tf.shape(e_t)[1]],tf.float32)-tf.reduce_sum(encoding_input*e_t,axis=-1)

    return cosd

def compute_sq_distance(a, b, state_step_sizes):
    result = 0.0
    for i in range(a.shape[-1]):
        # compute difference
        diff = a[..., i] - b[..., i]
        # wrap angle for theta
        if i == 2:
            diff = wrap_angle(diff)
        # add up scaled squared distance
        result += (diff / state_step_sizes[i]) ** 2
    return result


def atan2(x, y, epsilon=1.0e-12):
    """
    A hack until the tf developers implement a function that can find the angle from an x and y co-
    ordinate.
    :param x:
    :param epsilon:
    :return:
    """
    # Add a small number to all zeros, to avoid division by zero:
    x = tf.where(tf.equal(x, 0.0), x + epsilon, x)
    y = tf.where(tf.equal(y, 0.0), y + epsilon, y)

    angle = tf.where(tf.greater(x, 0.0), tf.atan(y / x), tf.zeros_like(x))
    angle = tf.where(tf.logical_and(tf.less(x, 0.0), tf.greater_equal(y, 0.0)), tf.atan(y / x) + np.pi, angle)
    angle = tf.where(tf.logical_and(tf.less(x, 0.0), tf.less(y, 0.0)), tf.atan(y / x) - np.pi, angle)
    angle = tf.where(tf.logical_and(tf.equal(x, 0.0), tf.greater(y, 0.0)), 0.5 * np.pi * tf.ones_like(x), angle)
    angle = tf.where(tf.logical_and(tf.equal(x, 0.0), tf.less(y, 0.0)), -0.5 * np.pi * tf.ones_like(x), angle)
    angle = tf.where(tf.logical_and(tf.equal(x, 0.0), tf.equal(y, 0.0)), tf.zeros_like(x), angle)
    return angle
