#!/bin/env python

import cv2
import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return = tf.Variable(initial)

def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

class Network:
    def __init__(self,width,height):
        self.learning_rate=1e-4
        self.w = width
        self.h = height
        self.global_step = tf.Variable(0,trainable=False,name='global_step')

        self.x = tf.Variable(tf.float,[None,784],name='Input x')
        self.label = tf.Variable(tf.float,[None,10],name='Label')

        self.x_flat = tf.reshape(self.x, [-1,28,28,1])
        #layer 1
        self.w_conv1 = weight_variable([5,5,1,32])
        self.b_conv1 = bias_variable([32])
        self.h_conv1 = tf.nn.relu(conv2d(self.x_flat, self.w_conv1) + self.b_conv1)
        self.p_conv1 = max_pool_2x2(self.h_conv1)

        #layer 2
        self.w_conv2 = weight_variable([5,5,32,64])
        self.b_conv2 = bias_variable([64])
        self.h_conv2 = tf.nn.relu(conv2d(self.p_conv1, self.w_conv2)+self.b_conv2)
        self.p_conv2 = max_pool_2x2(self.h_conv2)

        #flat for full connection
        self.p_conv2_flat = tf.reshape(self.p_conv2,[-1, 7*7*64])

        #layer 3
        self.w_fc1 = weight_variable([7*7*64, 1024])
        self.b_fc1 = bias_variable([1024])
        self.h_fc1 = tf.nn.relu(tf.matmul(self.p_conv2_flat, self.w_fc1)+self.b_fc1)
        
        #layer 4
        self.w_fc2 = weight_variable([1024,10])
        self.b_fc2 = bias_variable([10])
        self.y = tf.nn.softmax(tf.matmul(self.h_fc1, self.w_fc2)+self.b_fc2)

        self.loss = -tf.reduce_sum(self.label*log(self.y))
        self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

        predict = tf.equal(tf.argmax(self.y,1), tf.argmax(self.label,1))
        self.accuracy = tf.reduce_mean(tf.cast(predict,'float'))

        tf.summary.histogram('Weight',self.w_fc2)
        tf.summary.histogram('Bias',self.b_fc2)
        tf.summary.scalar('loss',self.loss)
        tf.summary.scalar('accuracy',self.accuracy)
