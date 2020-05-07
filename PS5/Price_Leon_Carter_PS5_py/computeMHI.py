# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 11:25:44 2019

@author: Carter
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
#import pdb
import os
from skimage.measure import compare_ssim


def computeMHI(directory): 
    subdirname = directory + '/'
    subdir = os.listdir(subdirname)
	
    MHI = []
    for seqnum in range(len(subdir)):
        # cycle through all sequences for this action category
       
        depthfiles = glob.glob(subdirname + subdir[seqnum] + '/' + '*.pgm');
        depthfiles = np.sort(depthfiles)
        D = []
        D1 = []
#        backSub = cv2.createBackgroundSubtractorKNN()
        for i in range(len(depthfiles)):
            depth = cv2.imread(depthfiles[i])
            depth = depth[:,:,0]
            d_thresh = 160
            binary = np.where(depth<d_thresh,1,0)
            binary = np.asarray(binary,dtype = 'float32')
#            edges = cv2.Canny(binary,100,200,7)
#            edges = np.where(edges>0,1,0)
            D1.append(binary)
            if i >0:
                diff = cv2.subtract(binary,background)
#                diff = np.where(diff<0,0,1)
                background = binary
#                diff = backSub.apply(D1[i])
#                (score, diff) = compare_ssim(D1[i], D1[i-1], full=True)
#                max_val = np.max(diff)
#                diff = (diff/max_val).astype("uint8")
                D.append(diff)
#                plt.imshow(diff)
            else:
                background = binary
            
        #motion Energy Image
#        E = []
#        Tau = 5
#        for i in range(5,len(D)):
#            temp = []
#            for j in range(len(Tau)):
#                temp = []
#                temp.append(D[i-j])
#            E.append(temp)
                 
        #motion History Image
        tau = len(D)-1
#        tau = 20
        H = []
        H.append(D[0])
        for i in range(1,len(D)):
            current = D[i]
            prev = H[i-1]
            temp = np.where(current == 1, tau, prev-1)
            temp = np.where(temp<0,0,temp)
            H.append(temp)
    
        MHI.append(H[tau])
        
    return MHI
    
    
    
BA_MHI = computeMHI('PS5_Data/PS5_Data/botharms')

directory = 'PS5_Data/PS5_Data/botharms'