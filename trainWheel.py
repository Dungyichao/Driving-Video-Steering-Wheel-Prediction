import tensorflow as tf
import ReadImage  # get image / get wheel value
from wheel_model import wheel_model
import tensorflow.contrib.slim as slim

class wh():
    def __init__(self):
        self.input = tf.placeholder(shape=[None, 66, 200, 3], dtype=tf.float32)
        #self.yreal = tf.placeholder(shape=[None, None], dtype=tf.float32)  # for real wheel value
        # self.conv1 = slim.conv2d(inputs=self.input, num_outputs=24, kernel_size=[5,5], stride=[2,2], padding='SAME', biases_initializer=None)
        self.W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 3, 24], stddev=0.1))
        self.b_conv1 = tf.Variable(tf.constant(0.1, shape=[24]))
        self.h_conv1 = tf.nn.relu(tf.nn.conv2d(self.input, self.W_conv1, strides=[1, 2, 2, 1], padding='VALID') + self.b_conv1)

#input = tf.placeholder(shape=[None,66,200,3], dtype=tf.float32)
#yreal = tf.placeholder(shape=[None,None], dtype=tf.float32)   # for real wheel value
        #self.conv1 = slim.conv2d(inputs=self.input, num_outputs=24, kernel_size=[5,5], stride=[2,2], padding='SAME', biases_initializer=None)
#W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 3, 24], stddev=0.1))
#b_conv1 = tf.Variable(tf.constant(0.1, shape=[24]))
#h_conv1 = tf.nn.relu(tf.nn.conv2d(input, W_conv1, strides=[1, 2, 2, 1], padding='VALID') + b_conv1)
tf.reset_default_graph()
wheel = wheel_model()
#who = wh()
init = tf.global_variables_initializer()
data = ReadImage.call_ep_imgdata(1)
steer = ReadImage.steer_value(1)
#print(steer)



with tf.Session() as sess:
    sess.run(init)
    for i in range(500):
        _ = sess.run(wheel.update_model, feed_dict={wheel.input: data, wheel.yreal:steer, wheel.keep_prob: 0.8})
        q = sess.run(wheel.loss, feed_dict={wheel.input: data, wheel.yreal: steer, wheel.keep_prob: 0.8})
        print("%d iteration loss: " % i, q)