#!/bin/env python

import PIL.Image as Image
import numpy as np
import cv2

#convert cv2 img to PIL Image
def cvImg2PilImg(cvimg):
    img = Image.frombytes('L',(cvimg.shape),cvimg.tostring())
    return img
#convert PIL img to numpy array
def PilImg2arr(im):
    img = np.array(im)
    return img
#convert arr to cv2 img
def arr2cvimg(arr):
    img = cv2.fromarray(arr)
    return img

def cvimg2arr(img):
    im = cvImg2PilImg(img)
    im2 = PilImg2arr(im)
    return im2

img = cv2.imread('test_images/3_pred_28.png',0)
im = cvImg2PilImg(img)
print(im)
arr = PilImg2arr(im)
print(arr)

