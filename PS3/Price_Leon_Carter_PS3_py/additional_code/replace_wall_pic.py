# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 23:18:16 2019

@author: Carter
"""

#bike_beach
from saveImage_inside_image import warpImage
from computeH import computeH
import cv2
import numpy as np
from get_matching_points_UserInput import get_corresponding_points

#do the same for the kendalpics
image1 = cv2.imread('bike_beach.jpg')
image2 = cv2.imread('wall_pic.jpg')
get_corresponding_points("bike_beach.jpg", "wall_pic.jpg")

t1 =[[0,0,],
      [image1.shape[1],0],
      [image1.shape[1],image1.shape[0]],
      [0,image1.shape[0]]]

t2 =[[299,136],
      [485,111],
      [478,305],
      [301,273]]


t1 = np.asarray(t1,dtype = 'float')
t2 = np.asarray(t2,dtype = 'float')

##save the points
#np.save('points1.npy',t1)
#np.save('points2.npy',t2)
#
##reload to verify
#t1_test = np.load('points1.npy')
#t2_test = np.load('points2.npy')

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

def scale_up(items, image):
    for i in range(len(items)):
        items[i,0] = items[i,0]/2*image.shape[0]
        items[i,1] = items[i,1]/2*image.shape[1]
    return items

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
result,merged = warpImage(image1,image2,h_2)

cv2.imwrite("wall_pic_result.jpg", result)
cv2.imwrite("merged_wall_pic_beach_bike.jpg", merged) 