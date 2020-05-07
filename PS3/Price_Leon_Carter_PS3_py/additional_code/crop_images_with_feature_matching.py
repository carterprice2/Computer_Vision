# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 16:32:31 2019

@author: Carter
"""

import cv2
import numpy as np
from computeH import computeH

#test on Crop images using the given points

#load the numpy arrays
image1 = cv2.imread('wdc1.jpg')
image2 = cv2.imread('wdc2.jpg')

orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(image1,None)
kp2, des2 = orb.detectAndCompute(image2,None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
img3 = cv2.drawMatches(image1,kp1,image2,kp2,matches[:10],None, flags=2)

cv2.imshow("matches", img3)
cv2.waitKey()
cv2.destroyAllWindows()

cv2.imwrite("ORB_wdc_matches.jpg",img3)
j = 0 
good = []
for m in matches:
        good.append(m)
        j = j +1
        if j >10:
            break

src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ])
dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ])
 
  
cc1 = src_pts
cc1 = np.asarray(cc1)
cc1 = cc1.reshape((11,2))

cc2 = np.asarray(dst_pts)
cc2 = cc2.reshape((11,2))

cc1_copy = np.copy(cc1)
cc2_copy = np.copy(cc2)

#cc1_rev = cc1
#cc2_rev = cc2
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
cc1_down_t  = cc1_down.T
cc2_down_t = cc2_down.T
h_2 = computeH(cc1_down_t,cc2_down_t)
#==========================================================================================================
##verify that the homography is correct
##using the h from computeH2
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

cv2.imwrite("ORB_wdc_verify.jpg",crops)
#================================================================================================================
#test warp image 
#test the warp image function
from warpImage import warpImage

image1 = cv2.imread('wdc1.jpg')
image2 = cv2.imread('wdc2.jpg')
result,large = warpImage(image1,image2,h_2)

cv2.imwrite("wdc_result_ORB_Features.jpg", result)
cv2.imwrite("wdc_merged_ORB_features.jpg", large) 