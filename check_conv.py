import tensorflow as tf
import numpy as np

# params for lrn
depth_radius = 5
bias = 1.0
alpha = 0.0001
beta = 0.75

# params for max_pool
ksize = 3
padding = 0
strides = 2

pool1_input = np.ones([1, 96, 55, 55], dtype=np.float32)
# print(pool1_input.dtype)

relu = tf.nn.relu(pool1_input)
lrn = tf.nn.lrn(relu, depth_radius, bias=bias, alpha=alpha, beta=beta)
print(lrn.shape)
# Transpose NCHW to NHWC
# lrn = tf.transpose(lrn, [0,2,3,1])
pool_out = tf.nn.max_pool2d(lrn, ksize=ksize, strides=strides, padding='VALID', data_format='NHWC', name='pool2')

with tf.compat.v1.Session() as sess:
    out = sess.run(lrn)
    for i in range(out.shape[1]):
        for j in range(out.shape[2]):
            for k in range(out.shape[3]):
                print(out[0][i][j][k])
