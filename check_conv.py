import tensorflow as tf
import numpy as np

# params for lrn
depth_radius = 2
bias = 1.0
alpha = 0.001 / 9.0
beta = 0.75

# params for max_pool
ksize = 3
padding = 0
strides = 2
input = np.array([
    [
        [230,230,230],[210,210,210],[190,190,190],[170,170,170],[150,150,150]
    ],
    [
        [230,230,230],[210,210,210],[190,190,190],[170,170,170],[150,150,150]
    ],
    [
        [230,230,230],[210,210,210],[190,190,190],[170,170,170],[150,150,150]
    ],
    [
        [230,230,230],[210,210,210],[190,190,190],[170,170,170],[150,150,150]
    ],
    [
        [230,230,230],[210,210,210],[190,190,190],[170,170,170],[150,150,150]
    ],
], dtype=np.float32)
input_for_tf = np.zeros([1, input.shape[0], input.shape[1], input.shape[2]], dtype=np.float32)
input_for_tf[0] = input
pool1_input = input_for_tf
pool1_input = np.ones([1, 55, 55, 96], dtype=np.float32)
# pool1_input = np.ones([1, 96, 55, 55], dtype=np.float32)
# print(pool1_input.dtype)

relu = tf.nn.relu(pool1_input)
lrn = tf.nn.lrn(relu, depth_radius, bias=bias, alpha=alpha, beta=beta)
# Transpose NCHW to NHWC
# lrn = tf.transpose(lrn, [0,2,3,1])
# pool_out = tf.nn.max_pool2d(lrn, ksize=ksize, strides=strides, padding='VALID', data_format='NHWC', name='pool2')

with tf.compat.v1.Session() as sess:
    out = sess.run(lrn)
    print(out.shape)
    for i in range(out.shape[1]):
        for j in range(out.shape[2]):
            for k in range(out.shape[3]):
                print(out[0][i][j][k])
