import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from tensorflow.python.ops import init_ops

class wheel_model():
    def __init__(self):
        #self.i = i
        self.iw()
    def iw(self):
        self.input = tf.placeholder(shape=[None,66,200,3], dtype=tf.float32)
        self.yreal = tf.placeholder(shape=[None], dtype=tf.float32)   # for real wheel value
        h_conv1 = slim.conv2d(inputs=self.input, num_outputs=24, kernel_size=[5,5], stride=[2,2], padding='VALID', biases_initializer=None)
        self.h_conv1 = tf.nn.relu(h_conv1)
        h_conv2 = slim.conv2d(inputs=self.h_conv1, num_outputs=36, kernel_size=[5,5], stride=[2,2], padding='VALID', biases_initializer=None)
        self.h_conv2 = tf.nn.relu(h_conv2)
        h_conv3 = slim.conv2d(inputs=self.h_conv2, num_outputs=48, kernel_size=[5,5], stride=[2,2], padding='VALID', biases_initializer=None)
        self.h_conv3 = tf.nn.relu(h_conv3)
        h_conv4 = slim.conv2d(inputs=self.h_conv3, num_outputs=64, kernel_size=[3,3], stride=[1,1], padding='VALID', biases_initializer=None)
        self.h_conv4 = tf.nn.relu(h_conv4)
        h_conv5 = slim.conv2d(inputs=self.h_conv4, num_outputs=64, kernel_size=[3, 3], stride=[1, 1], padding='VALID',
                              biases_initializer=None)
        self.h_conv5 = tf.nn.relu(h_conv5)
        self.h_conv5_flat = tf.reshape(self.h_conv5, [-1, 1152])
        ############################################# Fully Connect
        self.keep_prob = tf.placeholder(tf.float32)
        self.h_fc1 = tf.nn.relu(slim.fully_connected(self.h_conv5_flat, 1164))
        self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)
        self.h_fc2 = tf.nn.relu(slim.fully_connected(self.h_fc1_drop, 100))
        self.h_fc2_drop = tf.nn.dropout(self.h_fc2, self.keep_prob)
        self.h_fc3 = tf.nn.relu(slim.fully_connected(self.h_fc2_drop, 50))
        self.h_fc3_drop = tf.nn.dropout(self.h_fc3, self.keep_prob)
        self.h_fc4 = tf.nn.relu(slim.fully_connected(self.h_fc3_drop, 10))
        self.h_fc4_drop = tf.nn.dropout(self.h_fc4, self.keep_prob)
        self.y = tf.multiply(tf.atan(slim.fully_connected(self.h_fc4_drop, 1)), 2)
        self.loss = tf.reduce_mean(tf.square(tf.subtract(self.y, self.yreal)))
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.update_model = self.trainer.minimize(self.loss)
