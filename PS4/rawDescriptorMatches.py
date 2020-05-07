# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 23:38:36 2019

@author: Carter
"""

#PS4 number 1

import numpy as np
import scipy.io
#import glob
from scipy import misc
import matplotlib.pyplot as plt
from displaySIFTPatches import displaySIFTPatches
from selectRegion import roipoly
from getPatchFromSIFTParameters import getPatchFromSIFTParameters
from skimage.color import rgb2gray
import matplotlib.cm as cm
import pylab as pl
#import pdb
#import cv2

# specific frame dir and siftdir
framesdir = 'frames/'
siftdir = 'sift/'


#load the images for this script
#im1 = misc.imread(framesdir + 'friends_0000000117.jpeg')
#im2 = misc.imread(framesdir + 'friends_0000000118.jpeg')
#plt.imshow(im1)
#plt.imshow(im2)

two_frames_data = scipy.io.loadmat('twoFrameData.mat')

im1 = two_frames_data['im1']
plt.imshow(im1)

im2 = two_frames_data['im2']
plt.imshow(im2)

im1_mat = {'positions': two_frames_data['positions1'], 'orients': two_frames_data['orients1'], 'scales': two_frames_data['scales1'], 'descriptors': two_frames_data['descriptors1']}
im2_mat = {'positions': two_frames_data['positions2'], 'orients': two_frames_data['orients2'], 'scales': two_frames_data['scales2'], 'descriptors': two_frames_data['descriptors2']}

#load the SIFT features files
#fname1 = siftdir + 'friends_0000000117.jpeg.mat'
#im1_mat = scipy.io.loadmat(fname1)
#
#fname2=siftdir + 'friends_0000000118.jpeg.mat'
#im2_mat = scipy.io.loadmat(fname2)


#def preview(image):
#    cv2.imshow("preview", image)
#    cv2.waitKey()
#    cv2.destroyAllWindows()
#    
#preview(im1)
#preview(im2)

#select a region within image 1 
print ('use the mouse to draw a polygon, right click to end it')
pl.imshow(im1)
MyROI = roipoly(roicolor='r')

#run this later___the blocking feature is not working in my IDE
Ind = MyROI.getIdx(im1, im1_mat['positions'])
print("features in region", len(Ind))

#from load data script - run this to view the SIFT features in image1

fig=plt.figure()
bx=fig.add_subplot(111)
bx.imshow(im1)
coners = displaySIFTPatches(im1_mat['positions'][Ind,:], im1_mat['scales'][Ind,:], im1_mat['orients'][Ind,:])

for j in range(len(coners)):
    bx.plot([coners[j][0][1], coners[j][1][1]], [coners[j][0][0], coners[j][1][0]], color='g', linestyle='-', linewidth=1)
    bx.plot([coners[j][1][1], coners[j][2][1]], [coners[j][1][0], coners[j][2][0]], color='g', linestyle='-', linewidth=1)
    bx.plot([coners[j][2][1], coners[j][3][1]], [coners[j][2][0], coners[j][3][0]], color='g', linestyle='-', linewidth=1)
    bx.plot([coners[j][3][1], coners[j][0][1]], [coners[j][3][0], coners[j][0][0]], color='g', linestyle='-', linewidth=1)
bx.set_xlim(0, im1.shape[1])
bx.set_ylim(0, im1.shape[0])  
plt.gca().invert_yaxis()
plt.show()    
#
## extract the image patch
#
#patch_num = 1
#img_patch = getPatchFromSIFTParameters(mat['positions'][patch_num,:], mat['scales'][patch_num], mat['orients'][patch_num], rgb2gray(im1))
#
#plt.imshow(img_patch,  cmap = cm.Greys_r)
#plt.show()
import math

def euclidean_dis(a,b):
    '''
    2 list of numbers 
    return the euclidean distance between the two
    '''
    if len(a) != len(b):
        print("lengths do not match")
        return -1
    else:
        total = 0
        for i in range(len(a)):
            total = total + (a[i]-b[i])**2
        return math.sqrt(total)

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
ROI_mat_des = im1_mat['descriptors'][Ind,:]
im2_des = im2_mat['descriptors']
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

fig=plt.figure()
bx=fig.add_subplot(111)
bx.imshow(im2)
coners = displaySIFTPatches(im2_mat['positions'][inds,:], im2_mat['scales'][inds,:], im2_mat['orients'][inds,:])

for j in range(len(coners)):
    bx.plot([coners[j][0][1], coners[j][1][1]], [coners[j][0][0], coners[j][1][0]], color='g', linestyle='-', linewidth=1)
    bx.plot([coners[j][1][1], coners[j][2][1]], [coners[j][1][0], coners[j][2][0]], color='g', linestyle='-', linewidth=1)
    bx.plot([coners[j][2][1], coners[j][3][1]], [coners[j][2][0], coners[j][3][0]], color='g', linestyle='-', linewidth=1)
    bx.plot([coners[j][3][1], coners[j][0][1]], [coners[j][3][0], coners[j][0][0]], color='g', linestyle='-', linewidth=1)
bx.set_xlim(0, im2.shape[1])
bx.set_ylim(0, im2.shape[0])  
plt.gca().invert_yaxis()
plt.show()
