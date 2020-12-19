import tflib as lib

import numpy as np
import tensorflow as tf


def Conv2D(name, inputs, filter, kernel_size, len):

    with tf.name_scope(name) as scope:

        #fan_in = 1 * kernel_size**2
        #fan_out = filter * kernel_size**2
        #filters_stdev = np.sqrt(4. / (fan_in + fan_out))
        filter_value = np.random.uniform(
                low= np.sqrt(3),
                high = np.sqrt(3),
                size=[kernel_size, len, 1, filter]
            ).astype('float32')

        filters = lib.param(name+".Filters", filter_value)
        result = tf.nn.conv2d(
            input=inputs,
            filter=filters,
            strides=[1, 1, 1, 1],
            padding="VALID",
        )

        _biases = lib.param(
            name + '.Biases',
            np.zeros(filter, dtype='float32')
        )

        result = tf.nn.bias_add(result, _biases)
        result = tf.nn.tanh(result)
        return result
