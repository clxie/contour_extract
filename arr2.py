#!/bin/env python
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import os
import read_data
import time
import shutil

p = '/mnt/pc-hf198/chunleixie/ml/mnist/all/test2'

im1path = '/mnt/pc-hf198/chunleixie/ml/mnist/all/train/imgs/Page_000001_DefectID79491.tif'
im2path = '/mnt/pc-hf198/chunleixie/ml/mnist/all/train/label/Page_000001_DefectID79491.tif'

im_sz=10000
im_w=100
im_h=100

def calc_all(xs,labels):
    accs=[]
    (cnt,w) = xs.shape[:2]
    (cnt2,w2) = labels.shape[:2]
    for i in range(cnt):
        acc = calc_accuracy(xs[i],labels[i])
        accs.append(acc)
        print('---------> accuracy : %g' % (acc))

def calc_accuracy(x,label):
    cnt = 0
    for i in range(10000):
        if x[i] == label[i]:
            cnt += 1
    return (cnt/10000)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

y = tf.placeholder('float32',[None,10000],name='logits')
label = tf.placeholder('float32',[None,10000],name='label')
predict = tf.reduce_sum(tf.cast(tf.equal(label, (y)),'float'), 1)
accuracy = tf.reduce_mean(predict/10000)

x,label1 = read_data.read_img_label_data('/mnt/pc-hf198/chunleixie/ml/mnist/all/test2')
x1 = read_data.imgthres1d(x, 128, 1, 0)
#label1 = read_data.read_img_data(im2path)

print('x1 shape:',x1.shape)
print('label1 shape:',label1.shape)
tf_accuracy = sess.run(accuracy, feed_dict={y:x1,label:label1})
my_acc = calc_all(x1, label1)

print('tf accuracy: %g,' % (tf_accuracy))
