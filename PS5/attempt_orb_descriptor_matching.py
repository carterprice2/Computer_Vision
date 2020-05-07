# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 13:17:43 2019

@author: Carter
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math

MHIs = np.load('allMHI_2.npy')
#orb descriptor classification

# Initiate ORB detector
orb = cv.ORB_create()

descriptors = []
for i in range(len(MHIs)):
    im1 = MHIs[i,:,:]
    im1 = np.asarray(im1, dtype = 'uint8')
    
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(im1,None)

    descriptors.append(des1)

def euclidean_dis_np(a,b):
    if len(a) != len(b):
        print("lengths do not match")
        return -1
    else:
        a = np.asarray(a)
        b = np.asarray(b)
        c = np.subtract(a,b)
        c = c**2
        total = np.sum(c)
        return math.sqrt(total)
    
#compare descriptors from region to all descriptors in image 2
#scores returns the best 'match' in image 2 for each feature in image 1
ROI_mat_des = descriptors[4]
im2_des = descriptors[0]
scores = []
for j in range(len(ROI_mat_des)):
    top_score = float('inf')
    index = -1
    for i in range(len(im2_des)):
        diff = euclidean_dis_np(ROI_mat_des[j],im2_des[i])
        if diff < top_score:
            top_score = diff
            index = i
    scores.append([j,index,top_score])

scores_np = np.asarray(scores,dtype='uint32')
inds = scores_np[:,1]
inds = list(inds)
