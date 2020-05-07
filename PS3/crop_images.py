# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 16:32:31 2019

@author: Carter
"""
from computeH2 import computeH2
import cv2
import numpy as np
from computeH import computeH

#test on Crop images using the given points

#load the numpy arrays
cc1 = np.load("cc1.npy")
cc2 = np.load('cc2.npy')
image1 = cv2.imread('crop1.jpg')
image2 = cv2.imread('crop2.jpg')

cc1_copy = np.copy(cc1)
cc2_copy = np.copy(cc2)

#moved to the computeH2 function
cc1_rev = np.empty(cc1.shape)
for i in range(len(cc1)):
    cc1_rev[i,0] = cc1[i,1] 
    cc1_rev[i,1] = cc1[i,0]

cc2_rev = np.empty(cc2.shape)
for i in range(len(cc2)):
    cc2_rev[i,0] = cc2[i,1] 
    cc2_rev[i,1] = cc2[i,0]   
    
#view the given points in the images
for i in range(len(cc1)):
    cv2.circle(image1,(int(cc1[i,0]),int(cc1[i,1])),4,(255,0,0),-1)
for i in range(len(cc2)):
    cv2.circle(image2,(int(cc2[i,0]),int(cc2[i,1])),4,(255,0,0),-1)  
  
cv2.imshow("crop 1", image1)
cv2.waitKey()
cv2.imshow("crop 2", image2)
cv2.waitKey()
cv2.destroyAllWindows()

cv2.imwrite("crops_1_keypoints.jpg", image1)
cv2.imwrite("crops_2_keypoints.jpg", image2)

def scale_down(lis,image):
    for i in range(len(lis)):
        lis[i,0] = lis[i,0]/image.shape[0]*2
        lis[i,1] = lis[i,1]/image.shape[1]*2
    return lis

def scale_up(items, image):
    for i in range(len(items)):
        items[i,0] = items[i,0]/2*image.shape[0]
        items[i,1] = items[i,1]/2*image.shape[1]
    return items

cc1_down = scale_down(cc1_rev,image1)
cc2_down = scale_down(cc2_rev,image2)
cc1_down = cc1_down.T
cc2_down = cc2_down.T
h_2 = computeH(cc1_down,cc2_down)
#==========================================================================================================
##verify that the homography is correct
##using the h from computeH2
cc1_down = cc1_down.T
cc2_down = cc2_down.T
new_point_list = []
for i in range(len(cc1)):
    point = np.array([[cc1_down[i,0]],[cc1_down[i,1]],[1]])
    new_point = np.dot(h_2,point)
    new_point_list.append(new_point)

#convert from h result to x,y and return to correct scale
resulting_points = []
for point in new_point_list:
    x = point[0]/point[2]
    y = point[1]/point[2]
    x = x/2*image2.shape[0]
    y = y/2*image2.shape[1]
    resulting_points.append((x,y))

#display the points 
crops = image2.copy()
for i in range(len(cc2)):
    cv2.circle(crops,(int(cc2_copy[i,0]),int(cc2_copy[i,1])),10,(255,0,0),2)
    
for point in resulting_points:    
    cv2.circle(crops,(int(point[1][0]),int(point[0][0])),3,(0,0,255),-1)
    
cv2.imshow("crop 2", crops)
cv2.waitKey()
cv2.destroyAllWindows()
#================================================================================================================
#test warp image 
#test the warp image function
from warpImage import warpImage
result,failures = warpImage(image1,image2,h_2)

cv2.imwrite("crop_result_1.jpg", result)
cv2.imwrite("merged_crops.jpg", failures) 