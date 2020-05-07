# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 00:07:31 2019

@author: Carter
"""


from computeH import computeH
import cv2
import numpy as np
from get_matching_points_UserInput import get_corresponding_points

#do the same for the kendalpics
image1 = cv2.imread('pattern.jpg')
#get_corresponding_points("bike_beach.jpg", "pattern.jpg")

t1 =[[206,227],
      [248,190],
      [417,194],
      [450,231]]

a = 400

t2 =[[0,0],
      [a,0],
      [a,a],
      [0,a]]

t1 = np.asarray(t1,dtype = 'float')
t2 = np.asarray(t2,dtype = 'float')

image2 = np.zeros((int(t2[2,1]),int(t2[2,0]),3))

t1_copy = np.copy(t1)
t2_copy = np.copy(t2)

cc1_rev = np.empty(t1.shape)
for i in range(len(t1)):
    cc1_rev[i,0] = t1[i,1] 
    cc1_rev[i,1] = t1[i,0]

cc2_rev = np.empty(t2.shape)
for i in range(len(t2)):
    cc2_rev[i,0] = t2[i,1] 
    cc2_rev[i,1] = t2[i,0]   
    
#view the given points in the images
for i in range(len(t1)):
    cv2.circle(image1,(int(t1[i,0]),int(t1[i,1])),8,(255,0,0),-1)
for i in range(len(t2)):
    cv2.circle(image2,(int(t2[i,0]),int(t2[i,1])),8,(255,0,0),-1)  
  
cv2.imshow("im 1", image1)
cv2.waitKey()
cv2.imshow("im 2", image2)
cv2.waitKey()
cv2.destroyAllWindows()

def scale_down(lis,image):
    for i in range(len(lis)):
        lis[i,0] = lis[i,0]/image.shape[0]*2
        lis[i,1] = lis[i,1]/image.shape[1]*2
    return lis



t1_down = scale_down(cc1_rev,image1)
t2_down = scale_down(cc2_rev,image2)
h_2 = computeH(t1_down,t2_down)
#==========================================================================================================
##verify that the homography is correct
##using the h from computeH2
new_point_list = []
for i in range(len(t1)):
    point = np.array([[t1_down[i,0]],[t1_down[i,1]],[1]])
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
for i in range(len(t2_copy)):
    cv2.circle(crops,(int(t2_copy[i,0]),int(t2_copy[i,1])),10,(255,0,0),2)
    
for point in resulting_points:    
    cv2.circle(crops,(int(point[1][0]),int(point[0][0])),3,(0,0,255),-1)
    
cv2.imshow("crop 2", crops)
cv2.waitKey()
cv2.destroyAllWindows()
#================================================================================================================
#test warp image 
#test the warp image function
from side_2_orthogonal import warpImage
result = warpImage(image1,h_2,t2_copy)

cv2.imwrite("pattern_orthogonal.jpg", result)