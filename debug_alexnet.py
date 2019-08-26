import cv2
import numpy as np
import tensorflow as tf
import pdb

imagenet_mean = imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)
img_file_path = "./images/zebra.jpeg"

from alexnet import AlexNet
from caffe_classes import class_names

def run(img_path=img_file_path):
    #placeholder for input and dropout rate
    x = tf.placeholder(tf.float32, [1, 227, 227, 3])
    keep_prob = tf.placeholder(tf.float32)
    
    #create model with default config ( == no skip_layer and 1000 units in the last layer)
    model = AlexNet(x, keep_prob, 1000, [])
    
    #define activation of last layer as score
    score = model.fc8
    
    #create op to calculate softmax 
    softmax = tf.nn.softmax(score)
    
    sess = tf.Session()
        
    # Load the pretrained weights into the model
    sess.run(model.load_initial_weights(sess))
    
    image = cv2.imread(img_path)
    # Convert image to float32 and resize to (227x227)
    img = cv2.resize(image.astype(np.float32), (227,227))
    
    # Subtract the ImageNet mean
    img -= imagenet_mean
    
    # Reshape as needed to feed into model
    img = img.reshape((1,227,227,3))
    print(img.dtype)
    
    # Run the session and calculate the class probability
    probs = sess.run(softmax, feed_dict={x: img, keep_prob: 1})
    
    # Get the class name of the class with the highest probability
    class_name = class_names[np.argmax(probs)]
    print("Class: " + class_name + ", probability: %.4f" %probs[0,np.argmax(probs)])

def img_to_binary(img_file_path=img_file_path, output_file='./input_img.bin'):
    img = cv2.imread(img_file_path) # load image order BGR
    img = cv2.resize(img, (227, 227))
    
    # Transpose original image from HWC to CHW
    img = img.transpose([2, 0, 1])
    flatten_img = img.flatten()
    byte_data = flatten_img.tobytes()
    with open(output_file, 'bw') as f:
        f.write(byte_data)

    # Check true value and binary data on my eyes
    # output_value_file = './out_value.txt'
    # with open(output_value_file, 'w') as f:
    #     for v in flatten_img:
    #         f.write("{}\n".format(v))

    # Check byte data is same to the img data
    with open(output_file, 'br') as f:
        restored_data = f.read()
    restored_array = np.frombuffer(restored_data, dtype=np.uint8)

    print(np.array_equal(flatten_img, restored_array))


def load_weights(op_name, weights_path='./bvlc_alexnet.npy'):
    weights_dict = np.load(weights_path, allow_pickle=True, encoding='bytes').item()
    return weights_dict[op_name]



def check_first_conv(img_file_path, result_byte_data):
    # Restore binary data file to numpy array
    # with open(result_byte_data, 'br') as f:
    #     restored_data = f.read()
    # restored_result = np.frombuffer(restored_data, dtype=np.float32)

    # Load image
    img = cv2.imread(img_file_path)
    img = cv2.resize(img.astype(np.float32), (227, 227))
    conv_input = img.reshape((1,227,227,3))

    # Load weight and bias
    weights = load_weights(op_name='conv1')[0]
    bias = load_weights(op_name='conv1')[1]
    stride = 4
    output = tf.nn.conv2d(conv_input, filter=weights, strides=stride, padding='VALID', name='test_conv')
    output = tf.nn.bias_add(output, bias=bias)
    # output = tf.nn.conv2d(conv_input, filter=weights, strides=stride, padding='VALID', data_format='NHWC', dilations=[1, 1, 1, 1], name='test_conv')
    with tf.Session() as sess:
        out = sess.run(output)
        print(out[0][0][0][0])
        # print(np.array_equal(out, restored_result))

    # Write byte data for cheking FPGA result
    out = out.transpose([0,3,1,2]).flatten()
    print(out[0])
    byte_data = out.tobytes()
    with open('./tf_conv1_result.dat', 'bw') as f:
        f.write(byte_data)


if __name__ == "__main__":
    img_to_binary('./images/zebra.jpeg')
    # run(img_path='./images/airplane.jpeg')
    # check_first_conv('./images/zebra.jpeg', './output.dat')
