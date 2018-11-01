#!/bin/env python
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import os
import read_data
import time

imgpath = '/mnt/pc-hf198/chunleixie/ml/mnist/all/test/imgs'
imgpath2 = '/mnt/pc-hf198/chunleixie/ml/mnist/all/test/label'

read_data.show_result(imgpath, imgpath2)
