#coding=utf-8
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from model import Network
import os
import time
import read_data
import shutil


im_w=100
im_h=100
im_sz=im_w*im_h

CKPT_DIR = 'ckpt'

def output_prediction(inpath,outpath,y):
    files = os.listdir(inpath)
    sz = len(files)
    print("Prediction pics: %d" % (sz))
    ##compare result diff
    for idx in range(1,sz):
        diff = read_data.compare_img_diff(y[0],y[idx])
        print('Image same percent: %g' % (diff))
    for i in range(sz):
        if files[i].find('.tif') > 0:
            out = outpath + '/' + files[i]
            inp = inpath + '/' + files[i]
            img = Image.open(inp).convert('L')
            w,h = img.size
            arr = y[i]
            arr = np.reshape(arr,[im_w,im_h])
            arr = read_data.imgthres2d(arr,1,255,0)

            #convert numpy array to Image
            im_predict = Image.fromarray(arr)
            # resize 
            im_predict_rs = im_predict.resize((w,h),Image.BOX)
            #save image
            im_predict_rs.save(out)
    
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
        else:
            raise FileNotFoundError("Not saved model yet")

    def predict2(self, imagedir, outdir):
        x = read_data.read_dir_imgs(imagedir)
        y = self.sess.run(self.net.y, feed_dict={self.net.x: x})
        print('x shape:',x.shape)
        print('y shape:',y.shape)
        output_prediction(imagedir,outdir, y)


t1 = time.time()
if __name__ == "__main__":
    app = Predict()
    inputdir = '/mnt/pc-hf198/chunleixie/ml/mnist/all/test/imgs'
    outputdir = '/mnt/pc-hf198/chunleixie/ml/mnist/all/test/label'
    if os.path.exists(outputdir):
        shutil.rmtree(outputdir)
    os.mkdir(outputdir)
    app.predict2(inputdir,outputdir)

t2 = time.time()
print("######### prediction time: %g" % (t2 - t1))
