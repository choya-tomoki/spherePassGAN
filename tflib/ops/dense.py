import tflib as lib

import numpy as np
import tensorflow as tf


def Dense(name, inputs, dim):
    """
    inputs: tensor of shape (batch size, num channels, height, width)
    mask_type: one of None, 'a', 'b'

    returns: tensor of shape (batch size, num channels, height, width)
    """
    with tf.name_scope(name) as scope:

        result = tf.contrib.layers.fully_connected(
            inputs,
            dim,
            weights_regularizer=None,
            biases_regularizer=None,
            activation_fn=tf.nn.sigmoid)

        return result
