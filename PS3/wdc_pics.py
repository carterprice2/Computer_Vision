# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 19:41:53 2019

@author: Carter
"""


from computeH import computeH
import cv2
import numpy as np
from get_matching_points_UserInput import get_corresponding_points


#test on Crop images using the given points

#load the numpy arrays
image1 = cv2.imread('wdc1.jpg')
image2 = cv2.imread('wdc2.jpg')
get_corresponding_points("wdc1.jpg", "wdc2.jpg")

t1 =[[120,209],
[242,138],
[215,189],
[369,243],
[290,78],
[94,179],
[209,116],
[219,219],
[252,196],
[256,284],
[301,93],
[341,119],
[203,223],
[325,229]]

t2 = [[322,37],
[305,158],
[271,95],
[111,130],
[342,306],
[373,45],
[365,169],
[243,74],
[235,108],
[175,52],
[304,286],
[226,270],
[253,65],
[156,119]]

t1 = np.asarray(t1,dtype = 'float')
t2 = np.asarray(t2,dtype = 'float')

#save the points
t1 = t1.T
t2 = t2.T
np.save('points1.npy',t1)
np.save('points2.npy',t2)

#reload to verify
t1_test = np.load('points1.npy')
t2_test = np.load('points2.npy')
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
    cv2.circle(image1,(int(t1[i,0]),int(t1[i,1])),4,(255,0,0),-1)
for i in range(len(t2)):
    cv2.circle(image2,(int(t2[i,0]),int(t2[i,1])),4,(255,0,0),-1)  
  
cv2.imshow("im 1", image1)
cv2.waitKey()
cv2.imshow("im 2", image2)
cv2.waitKey()
cv2.destroyAllWindows()

cv2.imwrite("WDC_1_keypoints.jpg", image1)
cv2.imwrite("WDC_2_keypoints.jpg", image2) 

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

t1_down = scale_down(cc1_rev,image1)
t2_down = scale_down(cc2_rev,image2)
t1_down  = t1_down.T
t2_down = t2_down.T
h_2 = computeH(t1_down,t2_down)
#==========================================================================================================
##verify that the homography is correct
##using the h from computeH2
t1_down  = t1_down.T
t2_down = t2_down.T
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
from warpImage import warpImage
result,merged = warpImage(image1,image2,h_2)

cv2.imwrite("WDC_result_1.jpg", result)
cv2.imwrite("merged_WDC.jpg", merged) 