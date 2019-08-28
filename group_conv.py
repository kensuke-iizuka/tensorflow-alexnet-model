import tensorflow as tf
import numpy as np

def conv(x, weights, biases, stride_y, stride_x, name, padding='SAME', groups=1):
    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k, strides=[1, stride_y, stride_x, 1],
                                         padding=padding)

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
# Initialize input data 
    conv_input = np.arange(5*5*4, dtype=np.float32)
    conv_input = conv_input.reshape([1, 4, 5, 5])
    tf_input = conv_input.transpose([0, 2, 3, 1]) # NCHW to NHWC
    print(tf_input.shape)
    weights = np.arange(3*3*2*6, dtype=np.float32)
    weights = weights.reshape([6, 2, 3, 3])
    tf_weights = weights.transpose([2, 3, 1, 0]) # H x W x ICH x OCN
    # print(tf_weights)
    bias = np.zeros([6,]))
    stride = 1
    padding = 0
    output = conv(tf_input, tf_weights, bias, 1, 1, padding='VALID', name='group_test_conv', groups=2)
    with tf.compat.v1.Session() as sess:
        out = sess.run(output)
        print(type(out))
        print(out.shape)
        result = out.transpose([3, 0, 1, 2]) # NHWC to NCHW
        print(result)
