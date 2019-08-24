import tensorflow as tf
import numpy as np

def conv(x, weights, biases, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding='SAME', groups=1):
    """Create a convolution layer.

    Adapted from: https://github.com/ethereon/caffe-tensorflow
    """
    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride_y, stride_x, 1],
                                         padding=padding)

    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
        weights = tf.get_variable('weights', shape=[filter_height,
                                                    filter_width,
                                                    input_channels/groups,
                                                    num_filters])
        biases = tf.get_variable('biases', shape=[num_filters])

    if groups == 1:
        conv = convolve(x, weights)

    # In the cases of multiple groups, split inputs & weights and
    else:
        # Split input and weights and convolve them separately
        input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
        weight_groups = tf.split(axis=3, num_or_size_splits=groups,
                                 value=weights)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

        # Concat the convolved output together again
        conv = tf.concat(axis=3, values=output_groups)

    # Add biases
    bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

    return bias


if __name__ == "__main__":
    input = tf.get_variable('input', shape=[8, 5, 5], initializer=tf.constant_initializer(1))
    weight = tf.get_variable('weight', shape=[2, 2, 4, 10], initializer=tf.constant_initializer(1))
    bias = tf.get_variable('bias', shape=[10], initializer=tf.constant_initializer(2))

    conv_result = conv(input, weight, bias, 3, 3, 10, 1, 1, groups=2, name='conv')

    sess = tf.compat.v1.Session()
    result = sess.run(conv_result)
    print(result[0][0][0])
