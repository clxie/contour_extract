#codeing=utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from model import Network
from PIL import Image
import numpy as np
import cv2
import os
import read_data
import time
import sys

CKPT_DIR = 'ckpt'

class Train:
    def __init__(self):
        t1 = time.time()
        self.net = Network()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.train_step =4000001
        t2 = time.time()
        print('Train initial done.Used Time: %g' % (t2 - t1))
        sys.stdout.flush()
        #self.data = input_data.read_data_sets('../data_set', one_hot=True)

    def train(self, datadir):
        batch_size = 50
        step = 0
        save_interval = 500
        saver = tf.train.Saver(max_to_keep=1)

        merged_summary_op = tf.summary.merge_all()
        merged_writer = tf.summary.FileWriter("./log", self.sess.graph)

        ckpt = tf.train.get_checkpoint_state(CKPT_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            step = self.sess.run(self.net.global_step)
            print('Continue from')
            print('        -> Minibatch update : ', step)
            sys.stdout.flush()

        t1 = time.time()
        #x,label = read_data.read_img_label_data('/mnt/pc-hf198/chunleixie/ml/mnist/all/train')
        #train_data_path = '/mnt/pc-hf198/chunleixie/ml/mnist/all/train'
        x,label = read_data.read_img_label_data(datadir)
        t2 = time.time()
        print('Read data time: %g' % (t2 - t1))
        sys.stdout.flush()

        while step < self.train_step:
            #x, label = self.data.train.next_batch(batch_size)
            x1,label1 = read_data.get_batch_data(x,label,batch_size,step)
            #_, loss, merged_summary = self.sess.run(
            #    [self.net.train, self.net.loss, merged_summary_op],
            #    feed_dict={self.net.x: x1, self.net.label: label1})
            _ = self.sess.run(self.net.train,
                feed_dict={self.net.x: x1, self.net.label: label1})

            step = self.sess.run(self.net.global_step)

            if self.train_step > 200:
                if step % 100 == 0:
                    loss_val,accuracy = self.sess.run([self.net.loss,self.net.accuracy], feed_dict={self.net.x: x1, self.net.label: label1})
                    print('loss : %g, accuracy: %g' % (loss_val,accuracy))
                    sys.stdout.flush()
            else:
                if step % 2 == 0:
                    loss_val,accuracy = self.sess.run([self.net.loss,self.net.accuracy], feed_dict={self.net.x: x1, self.net.label: label1})
                    print('loss : %g, accuracy: %g' % (loss_val,accuracy))
                    sys.stdout.flush()
                
            if step % save_interval == 0:
                merged_summary = self.sess.run(merged_summary_op, feed_dict={self.net.x:x1,self.net.label:label1})
                merged_writer.add_summary(merged_summary, step)
                saver.save(self.sess, CKPT_DIR + '/model', global_step=step)
                print('%s/model-%d saved' % (CKPT_DIR, step))
                sys.stdout.flush()

    #def calculate_accuracy(self):
    #    test_x = self.data.test.images
    #    test_label = self.data.test.labels
    #    accuracy = self.sess.run(self.net.accuracy,
    #                             feed_dict={self.net.x: test_x, self.net.label: test_label})
    #    print("accuracy = %g, length : %g" % (accuracy, len(test_label)))


t1 = time.time()    
if __name__ == "__main__":
    app = Train()
    datapath = '/mnt/pc-hf198/chunleixie/ml/mnist/all/train/'
    app.train(datapath)
t2 = time.time()

print("####### train step : %d, used time : %g " % (app.train_step, t2 - t1))
