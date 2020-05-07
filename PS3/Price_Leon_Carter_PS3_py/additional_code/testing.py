# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 09:29:47 2019

@author: Carter
"""
import numpy as np
import cv2
from get_matching_points_UserInput import get_corresponding_points

get_corresponding_points("crop1.jpg", "crop2.jpg")
#testing function file

#=====================================================================================================
#correspondence points for crop images
t1 = [[160,147,126,198,265,129,152,231,274,233,206,213,200,104],
      [148, 62,121,237,365, 70, 31,377,454,300,250,216,285,163]]
t2 = [[155,114,112,226,348, 98,110,323,410,290,237,234,250,99],
      [188,164,234,171,118,200,137,195,156,146,167,133,197,301]]

img1_shape = (319,480)
img2_shape = (450,317)


#convert to range 0-2
for i in range(len(t1[0])):
    t1[0][i] = t1[0][i]/img1_shape[0] *2
    t1[1][i] = t1[1][i]/img1_shape[1] *2
    
for i in range(len(t1[0])):
    t2[0][i] = t2[0][i]/img2_shape[0] *2
    t2[1][i] = t2[1][i]/img2_shape[1]*2
    
#output from crop images
#Point in image 1 (148, 160)
#point in image 2 (188, 155)
#Point in image 1 (62, 147)
#point in image 2 (164, 114)
#Point in image 1 (121, 126)
#point in image 2 (234, 112)
#Point in image 1 (237, 198)
#point in image 2 (171, 226)
#Point in image 1 (365, 265)
#point in image 2 (118, 348)
#Point in image 1 (70, 129)
#point in image 2 (200, 98)
    
#=========================================================================================================   
#verify that the homography is correct

#multiply the points by h matrix
#using the h from computeH
new_point_list = []
for i in range(len(t1[0])):
    point = np.array([[t1[0][i]],[t1[1][i]],[1]])
    new_point = np.dot(h,point)
    new_point_list.append(new_point)

#convert from h result to x,y and return to correct scale
resulting_points = []
for point in new_point_list:
    x = point[0]/point[2]
    y = point[1]/point[2]
    x = x/2*img2_shape[0]
    y = y/2*img2_shape[1]
    resulting_points.append((x,y))

#display the points 
crops = cv2.imread('crop2.jpg')
for i in range(len(t2[0])):
    cv2.circle(crops,(t2[1][i],t2[0][i]),10,(255,0,0),2)
    
for point in resulting_points:    
    cv2.circle(crops,(int(point[1][0]),int(point[0][0])),3,(0,0,255),-1)
    
cv2.imshow("crop 2", crops)
cv2.waitKey()
cv2.destroyAllWindows()

#=============================================================================================================
#test the warp image function
from warpImage import warpImage
inputimg = cv2.imread('crop1.jpg')
refImg = cv2.imread('crop2.jpg')
result,failures = warpImage(inputimg,refImg,h)

import cv2
cv2.imwrite("crop_result.jpg", result)
#=============================================================================================================
#test the warp image function
from warpImage2 import warpImage2
inputimg = cv2.imread('crop1.jpg')
refImg = cv2.imread('crop2.jpg')
result = warpImage2(inputimg,refImg,h)
#========================================================
#troubleshooting
#start with 4 points
inputIm = inputimg
H = h
output = refImg
point = np.array([[62/inputIm.shape[0]*2],[147/inputIm.shape[1]*2],[1]])
new_point = np.dot(H,point)
x = int((new_point[0]/new_point[2])/2*output.shape[0])
y = int(new_point[1]/new_point[2]/2*output.shape[1])
print(x,y)

#troubleshooting the inverse
test = np.zeros(inputIm.shape)
test = np.asarray(test,dtype = 'uint8')
for i in range(len(t2[0])):
    a = t2[0][i]
    b = t2[1][i]
    H_inv = np.linalg.inv(H)
    point = np.array([[0/output.shape[0]*2],[0/output.shape[1]*2],[1]])
    inv_point = np.dot(H_inv,point)
    x = int((inv_point[0]/inv_point[2])/2*inputIm.shape[0])
    y = int((inv_point[1]/inv_point[2])/2*inputIm.shape[1])
    cv2.circle(test,(x,y),3,(0,0,255),-1)
    print(x,y)
    
    
cv2.imshow("test", test)
cv2.waitKey()
cv2.destroyAllWindows()
#=================================================================================





