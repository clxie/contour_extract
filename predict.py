# -*- coding: UTF-8 -*-

import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from model import Network
import os
import time
import read_data
import shutil

'''
python 3.6
tensorflow 1.4
pillow(PIL) 4.3.0
'''
im_w=100
im_h=100
im_sz=im_w*im_h

CKPT_DIR = '/mnt/pc-hf197/chunleixie/ml/mnist/ckpt'

def merge_all_pics(y,img,curTotal):
    w,h = img.size
    r = h//im_h
    c = w//im_w

    for row in range(r+1):
        for col in range(c+1):
            if col == c:
                x0 = w - im_w
                x1 = w
            else:
                x0 = col * im_w
                x1 = (col+1) * im_w

            if row == r:
                y0 = h - im_h
                y1 = h
            else:
                y0 = row * im_h
                y1 = (row + 1) * im_h
            roi_box = (x0,y0,x1,y1)
            arr = y[curTotal]
            arr = np.reshape(arr,[im_w,im_h])
            arr = read_data.imgthres2d(arr, 1, 255, 0)
            i2 = Image.fromarray(arr)
            img.paste(i2,roi_box)
            curTotal += 1

    return (img,curTotal)

def output_prediction(inpath,outpath,y, method=0):
    files = os.listdir(inpath)
    sz = len(files)
    cnt,pix = y.shape
    print("Prediction pics: %d, count = %g, pix = %g" % (sz,cnt,pix))
    ##compare result diff
    #for idx in range(1,sz):
    #    diff = read_data.compare_img_diff(y[1],y[idx])
    #    print('Image same percent: %g' % (diff))
    if method == 0:
        for i in range(sz):
            #if files[i].find('.tif') > 0:
            out = outpath + '/' + files[i]
            inp = inpath + '/' + files[i]
            img = Image.open(inp).convert('L')
            w,h = img.size
            arr = y[i]
            arr = np.reshape(arr,[im_w,im_h])
            arr = read_data.imgthres2d(arr,1,255,0)

            #convert numpy array to Image
            im_predict = Image.fromarray(arr)
            im_predict = im_predict.convert('RGB')
            # resize 
            im_predict_rs = im_predict.resize((w,h),Image.BOX)
            #save image
            im_predict_rs.save(out)
    else:
        curTotal = 0
        for i in range(sz):
            out = outpath + '/' + files[i]
            inp = inpath + '/' + files[i]
            img = Image.open(inp).convert('L')
            merged_img,curTotal = merge_all_pics(y,img,curTotal)
            merged_rgb = merged_img.convert('RGB')
            merged_rgb.save(out)
    
class Predict:
    def __init__(self):
        self.net = Network()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.restore()

    def restore(self):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(CKPT_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            step = self.sess.run(self.net.global_step)
            print('Predict from step ---------> %g' % (step))
        else:
            raise FileNotFoundError("Not saved model yet")

    def predict(self, image_path):
        #img = Image.open(image_path).convert('L')
        #flatten_img = np.reshape(img, 784)
        #x = np.array([1 - flatten_img])
        #print('        -> Predict digit', np.argmax(y[0]))
        #x = read_dir_imgs(image_path)
        ##################################################################################
        #test one image 
        print(image_path)
        x = read_data.read_img_data(image_path)
        #img_x3 = read_data.img_process(x)
        y = self.sess.run(self.net.y, feed_dict={self.net.x: np.array([img_x3])})

        im = Image.open(image_path).convert('L')
        w,h = im.size
        arr = y[0]
        arr = read_data.imgthres1d(arr, 1, 255, 0)
        im_predict = Image.fromarray((np.reshape(arr,[im_w,im_h])))
        im_predict_rs = im_predict.resize((w,h),Image.BOX)
        im.show()
        im_predict_rs.show()
        ##################################################################################
        #test multiple images 
    def predict2(self, imagedir, outdir, method=0):
        x = read_data.read_dir_imgs(imagedir,method)
        y = self.sess.run(self.net.y, feed_dict={self.net.x: x})
        output_prediction(imagedir,outdir, y, method)


t1 = time.time()
if __name__ == "__main__":
    app = Predict()
    #app.predict('../test_images/0.png')
    #app.predict('../test_images/1.png')
    inputdir = '/mnt/pc-hf198/chunleixie/ml/mnist/all/test/imgs'
    outputdir = '/mnt/pc-hf198/chunleixie/ml/mnist/all/test/label'
    if os.path.exists(outputdir):
        shutil.rmtree(outputdir)
    os.mkdir(outputdir)
    app.predict2(inputdir,outputdir, 1)
    #app.predict('../test_images/4.png')

t2 = time.time()
print("######### prediction time: %g" % (t2 - t1))
