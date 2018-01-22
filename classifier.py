#!/usr/bin/env python
#encoding:utf-8
'''
    author: ISS=Kerui
    e-mail: e0267487@u.nus.edu
    date: 22 Jan 2018

    This function performs SVM classification.

'''
def overlapping_area(detection_1, detection_2):
from sklearn.svm import LinearSVC
from skimage.feature import hog
from sklearn.externals import joblib
import numpy as np
import glob
import os
import time
from skimage.io import imread
from skimage.transform import pyramid_gaussian
from nms import nms
import cv2
model_path = 'data/model/svm_train.model'
neg_feat_ph = 'data/features/train3/neg'
pos_feat_ph = 'data/features/train3/pos'
window_dw_sz = [20,20]
step_size = [5,5]
def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # 初始化缩放比例，并获取图像尺寸
    dim = None
    (h, w) = image.shape[:2]

    # If the width and height are 0, the original graph is returned.
    if width is None and height is None:
        return image

    if width is None:
       
        r = height / float(h)
        dim = (int(w * r), height)

    # if the height is zero
    else:
        # Scale the scaling according to the width
        r = width / float(w)
        dim = (width, int(h * r))

    # Resize picture
    resized = cv2.resize(image, dim, interpolation=inter)

    return resized
def sliding_window(image, window_size, step_size):
    
    for y in range(20, image.shape[0]-90, step_size[1]):
        for x in range(20, image.shape[1], step_size[0]):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

def train_classifer():
    t0 = time.time()
    fds = []
    labels = []
    # Load the positive features
    for feat_path in glob.glob(os.path.join(pos_feat_ph,"*.feat")):
        fd = joblib.load(feat_path)
        fds.append(fd)
        labels.append(1)
       

    # Load the negative features
    for feat_path in glob.glob(os.path.join(neg_feat_ph,"*.feat")):
        fd = joblib.load(feat_path)
        fds.append(fd)
        labels.append(0)
   
    clf = LinearSVC()
   
    clf.fit(fds, labels)
   # print clf.score(fds,labels)
    if not os.path.isdir(os.path.split(model_path)[0]):
        os.makedirs(os.path.split(model_path)[0])
    joblib.dump(clf, model_path)
    t1 = time.time()
    print ('The cast of time is :%f'%(t1-t0))
def test_classifer(img_path):
    scale = 0
    clf = joblib.load(model_path)
    detections = []
    im = imread(img_path, as_grey=True)
    im = resize(im,240,240)

    for (x, y, im_window) in sliding_window(im, window_dw_sz, step_size):
        
        if im_window.shape[0] != window_dw_sz[1] or im_window.shape[1] != window_dw_sz[0]:
            continue
        # Calculate the HOG features
        fd = hog(im_window, 12, (5,5), (2,2), visualise=False, transform_sqrt=True)
        pred = clf.predict([fd])
        if pred == 1 :
            #print  "Detection:: Location -> ({}, {})".format(x, y)

            detections.append((x, y, clf.decision_function([fd]),
                int(window_dw_sz[0]),
                int(window_dw_sz[1])))

    detections = nms(detections, 0.05)
    clone = im.copy()
    detections_copy = np.array(detections)
    mean_score = detections_copy.mean(axis = 0)[2]
   
    #Display the results after performing NMS
    for (x_tl, y_tl, score, w, h) in detections:
        if score>mean_score:
        # Draw the detections
            cv2.rectangle(clone, (x_tl, y_tl), (x_tl+w,y_tl+h), (255, 0, 0), thickness=1)
    cv2.imshow("Final Detections after applying NMS", clone)
    cv2.waitKey()


if __name__ == "__main__":
    #train_classifer()
    
    #test_classifer('2000.jpg')
    




