#!/bin/env python

import numpy as np
import PIL.Image as Image
import cv2
import os

im_w=100
im_h=100
im_c=2
im_sz=im_w*im_h

#convert cv2 img to PIL Image
def cvImg2PilImg(img):
    im = Image.frombytes('L',img.shape,img.tostring())
    return im

#convert PIL img to numpy array
def PilImg2arr(im):
    arr = np.array(im)
    return arr
#convert arr to cv2 img
def arr2cvimg(arr):
    pimg = Image.fromarray(arr) 
    pimg2 = pimg.convert('RGB')
    img = cv2.cvtColor(np.asarray(pimg2),cv2.COLOR_BGR2GRAY)
    return img

#convert cv2 img to numpy array
def cvimg2arr(img):
    im = cvImg2PilImg(img)
    arr = PilImg2arr(im)
    return arr

#input numpy array 
#output two arrs, one souce array, one canny array
def img_process(img):
    src = np.reshape(img,[im_w,im_h])
    im = arr2cvimg(src)
    imcanny = cv2.Canny(im,80,160)
    arr = []#np.random.random(im_w*im_h*im_c).reshape([28,28,2])
    arr.append(cvimg2arr(im))
    arr.append(cvimg2arr(imcanny))
    arr2 = np.reshape(arr,[2,im_w,im_h])
    return arr2

#input img path
#output numpy array
def read_img_data(path):
    img = Image.open(path).convert('L')
    img_sz = img.resize((im_w,im_h),Image.BOX)
    img_flat = np.reshape(img_sz, im_sz)

    return img_flat

#input img path
#output img array
def read_label_data(path):
    label = read_img_data(path)
    for i in range(img_sz):
        if label[i] > 128:
            label[i] = 1.0
        else:
            label[i] = 0.0
    label_flat = np.reshape(label, im_sz)

    return label_flat

#input img dir
#output img array[img number, img flattern array(w*h)]
def read_dir_imgs(path):
    filelist = os.listdir(path)
    img_data=[]
    for i in range(len(filelist)):
        img_path = path + '/' + filelist[i]
        if filelist[i].find('.tif') > 0:
            img = read_img_data(img_path)
            img_data.append(img)
    return img_data

def read_dir_labels(path):
    filelist = os.listdir(path)
    label_data=[]
    for i in range(len(filelist)):
        label_path = path + '/' + filelist[i]
        if os.path.isfile(img_path) & filelist[i].find('.tif') > 0:
            label = read_label_data(label_path)
            label_data.append(label)
    return label_data

#x = read_dir_imgs('../SEM_Class_page_1')
#a1 = x[0]
#a11 = np.reshape(a1, [im_w,im_h])
#img = Image.fromarray(a11)
#img_rz=img.resize((400,400),Image.BOX)
#img_rz.save('arr_test.tif')
#img_rz.show()
#
##########################################################################
# for test arr2 
#single_path = '../SEM_Class_page_1/Page_000001_DefectID9982.tif'

#img = read_img_data(single_path)

#arr2 = img_process(img)
#img1 = Image.fromarray(arr2[0,:,:])
#img2 = Image.fromarray(arr2[1,:,:])
#img1.show()
#img2.show()
##########################################################################

#imglists = os.listdir('SEM_Class_page_1')
#for i in range(len(imglists)):
#    if imglists[i].find('.tif') > 0:
#        path = 'SEM_Class_page_1/' + imglists[i];
#        outpath = 'label/' + imglists[i]
#        img = cv2.imread(path, 0)
#        h,w = img.shape
#        for r in range(h):
#            for c in range(w):
#                if img[r,c] > 128:
#                    img[r,c] = 255 
#                else:
#                    img[r,c] = 0
#        cv2.imwrite(outpath, img)

l = []
a = [[1,2,3],[4,5,6]]
l.append(a[:len(a)])
na = np.array(a)
nl = np.array(l)
print('a shape:',na.shape)
print('l shape:',nl.shape)
