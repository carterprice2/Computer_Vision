# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 17:14:12 2019

@author: Carter
"""

import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn.cluster import KMeans
from quantizeHSV import quantizeHSV
# [histEqual, histClustered] = getHueHists(im, k) 

def getHueHists(image,k):
        
    (h, w) = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # reshape the image into a feature vector so that k-means
    # can be applied
    im = hsv.reshape((hsv.shape[0] * hsv.shape[1], 3))
#    print(quantized_im.shape)
#    print(quant_im.shape)
#    print(cluster_centers)
    
#    cv2.imshow("quant im" , quantized_im)
#    cv2.waitKey()
#    cv2.destroyAllWindows()
    
    hue = im[:,0]
    hue = hue.reshape(-1,1)
    print(hue)
    
    #evently spaced bins
    evenHist = np.histogram(hue,k)
    plt.figure(1)
    n, bins, patches = plt.hist(hue, k)
    plt.title("Even Bins for k =" + str(k))
    plt.show()
    
    #get the centers
    quantizedHue, meanHues = quantizeHSV(image,k)
    
#    hsv2 = cv2.cvtColor(quantizedHue, cv2.COLOR_BGR2HSV) 
#    reshaped = hsv.reshape((hsv2.shape[0] * hsv2.shape[1], 3))
#    hue2 = reshaped[:,0]
#    hue2 = hue2.reshape(-1,1)
    
    #find the i cluster centers and make them the center of the bins
    meanHues = sorted(list(meanHues))
    bin_list = []
    bin_list.append(0)
    bin_list.append(np.max(hue))
    for i in range(len(meanHues)-1):
        div = int((meanHues[i] + meanHues[i+1])/2)
        bin_list.append(div)
     
    bin_list = sorted(bin_list)
      
    histClustered = np.histogram(hue, bin_list)
    plt.figure(2)
    n,bins, patches = plt.hist(hue, bin_list)
    plt.title("Clustered Bins for k =" + str(k))
    plt.show()
    
    return evenHist, histClustered


#    bin_list = list(meanHues)
#    first = (bin_list[0] + bin_list[1])/2
#    second = (bin_list[])
#    bin_list.append(0)
#    bin_list = sorted(bin_list)
#    new_bin_list = []
#    for item in bin_list:
#        print(item)
#        new_bin_list.append(int(np.round(item)))
#        
#        
#    #unevenly spaced bins based on mean hues
#    new_bin_list[3]= np.max(hue)
#    new_bin_list = [0, 69.5, 146,180]
#    histClusterd = np.histogram(hue2, new_bin_list)
#    n,bins, patches = plt.hist(hue2, new_bin_list)
#    plt.show()
#
#    bob = 0
#    for i in hue2:
#        if i < 50:
#            bob = bob + 1