# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 16:11:29 2019

@author: Carter
"""
from computeH import computeH
from warpImage import warpImage
import numpy as np
import cv2

def scale_down(lis,image):
    for i in range(len(lis)):
        lis[i,0] = lis[i,0]/image.shape[0]*2
        lis[i,1] = lis[i,1]/image.shape[1]*2
    return lis

def scale_down_2(t1,t2):
    max_val = max(np.max(t1),np.max(t2))
    t1 = t1/max_val
    t2 = t2/max_val
    return t1,t2

def reverse(cc1):
    cc1_rev = np.empty(cc1.shape)
    for i in range(len(cc1)):
        cc1_rev[i,0] = cc1[i,1] 
        cc1_rev[i,1] = cc1[i,0]
    return cc1_rev
    
t1 = np.load('points1.npy')
t2 = np.load('points2.npy')

img1 = cv2.imread('wdc1.jpg')
img2 = cv2.imread('wdc2.jpg')

t1,t2 = scale_down_2(t1,t2)
t1 = scale_down(t1.T,img1)
t2 = scale_down(t2.T,img2)

t1 = reverse(t1.T)
t2 = reverse(t2.T)
t1 = t1.T
t2 = t2.T

h = computeH(t1,t2)

final, merged = warpImage(img1,img2,h)

cv2.imshow("final", final)
cv2.waitKey()
cv2.imshow("merged", merged)
cv2.waitKey()
cv2.destroyAllWindows()
