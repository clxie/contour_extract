# -*- coding: UTF-8 -*-
import tensorflow as tf
import cv2
import numpy as np
import PIL.Image as Image
import read_data


#class Network:
#    def __init__(self):
#        self.learning_rate = 0.001
#        self.global_step = tf.Variable(0, trainable=False, name="global_step")
#
#        self.x = tf.placeholder(tf.float32, [None, 784], name="x")
#        self.label = tf.placeholder(tf.float32, [None, 10], name="label")
#
#        self.w = tf.Variable(tf.zeros([784, 10]), name="fc/weight")
#        self.b = tf.Variable(tf.zeros([10]), name="fc/bias")
#        self.y = tf.nn.softmax(tf.matmul(self.x, self.w) + self.b, name="y")
#
#        self.loss = -tf.reduce_sum(self.label * tf.log(self.y + 1e-10))
#        self.train = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(
#            self.loss, global_step=self.global_step)
#
#        predict = tf.equal(tf.argmax(self.label, 1), tf.argmax(self.y, 1))
#        self.accuracy = tf.reduce_mean(tf.cast(predict, "float"))
#
#        # 创建 summary node
#        # w, b 画直方图
#        # loss, accuracy画标量图
#        tf.summary.histogram('weight', self.w)
#        tf.summary.histogram('bias', self.b)
#        tf.summary.scalar('loss', self.loss)
#        tf.summary.scalar('accuracy', self.accuracy)

im_w=100
im_h=100
im_sz = im_w*im_h

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

class Network:
    def __init__(self):
        self.learning_rate = 1e-4 
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        
        self.x = tf.placeholder(tf.float32, [None, im_sz], name="x")
        self.label = tf.placeholder(tf.float32, [None, im_sz], name="label")
        
        #reshape 1D img to 3D
        self.x_img = tf.reshape(self.x,[-1, im_w, im_h, 1])

        #layer 1
        self.w_conv1 = weight_variable([3,3,1,32])
        self.b_conv1 = bias_variable([32])
        self.h_conv1 = tf.nn.relu(conv2d(self.x_img, self.w_conv1) + self.b_conv1)
        self.pool1 = max_pool_2x2(self.h_conv1)
        
        #layer 2
        self.w_conv2 = weight_variable([3,3,32,64])
        self.b_conv2 = bias_variable([64])
        self.h_conv2 = tf.nn.relu(conv2d(self.pool1, self.w_conv2) + self.b_conv2)
        self.pool2 = max_pool_2x2(self.h_conv2)

        #layer 3
        #self.w_conv3 = weight_variable([5,5,64,1])
        #self.b_conv3 = bias_variable([1])
        #self.output = tf.sigmoid(conv2d(self.h_conv2,self.w_conv3) + self.b_conv3)
        #layer 3 full connection
        self.w_fc = weight_variable([25*25*64, 10000])
        self.b_fc = bias_variable([10000])
        self.pool2_flat = tf.reshape(self.pool2, [-1,25*25*64])
        self.y = tf.sigmoid(tf.matmul(self.pool2_flat, self.w_fc) + self.b_fc)
        
        #reshape output to label shape
        #self.y = tf.reshape(self.output, [-1, im_sz])
         
        #predict = tf.equal(tf.argmax(self.label, 1), tf.argmax(self.y, 1))
        predict = tf.reduce_sum(tf.cast(tf.equal(self.label, self.y),'float'), 1)
        self.accuracy = tf.reduce_mean(predict/im_sz)

        #self.loss = tf.reduce_sum(tf.square(self.y - self.label)/2)
        self.loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=self.y))
        #self.loss = tf.nn.weighted_cross_entropy_with_logits(labels=self.label, logits=self.y)
        self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)
        
        
        # 创建 summary node
        # w, b 画直方图
        # loss, accuracy画标量图
        #tf.summary.histogram('weight', self.w_conv2)
        #tf.summary.histogram('bias', self.b_conv2)
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)
