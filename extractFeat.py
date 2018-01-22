#!/usr/bin/env python
#encoding:utf-8
'''
***************************************

author: ISS=Kerui
date: 22 Jan 2018
This function performs extracting feature maps woth HOG and storing those feature data.
***************************************
'''

from skimage.feature import hog
import numpy as np
from sklearn.externals import joblib
import time
import glob
from skimage.io import imread
import os
import cv2
from PIL import Image
neg_feat_ph = 'data/features/train/neg'
pos_feat_ph = 'data/features/train/pos'

def getData():
    
    
    for im_path in glob.glob(os.path.join('data/Train/posdata', "*.jpg")):
        # im = imread(im_path, as_grey=True)
        im = Image.open(im_path).convert('L')
        im = im.resize((20,20),Image.ANTIALIAS) 
        fd = hog(im, 12, (5,5), (2,2),visualise=False,transform_sqrt=True)
        fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
        fd_path = os.path.join(pos_feat_ph, fd_name)
        joblib.dump(fd, fd_path)
    print "Positive features saved in {}".format(pos_feat_ph)
    for im_path in glob.glob(os.path.join('data/Train/negdata', "*.jpg")):
        #im = imread(im_path, as_grey=True)
        im = Image.open(im_path).convert('L')
        im = im.resize((20,20),Image.ANTIALIAS) 
        fd = hog(im, 12, (5,5), (2,2),visualise=False,transform_sqrt=True)
        fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
        fd_path = os.path.join(neg_feat_ph, fd_name)
        joblib.dump(fd, fd_path)
    print "Negative features saved in {}".format(neg_feat_ph)
    


if __name__ == '__main__':
    
    getData()
    print "Features are extracted and saved."
  



