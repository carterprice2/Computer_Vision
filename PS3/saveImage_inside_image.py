# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 16:49:11 2019

@author: Carter
"""

import numpy as np
import cv2
import math


def bilinear_interp(x,y,image):
    r_x = math.floor(x)
    r_y = math.floor(y)
    a = x - r_x
    b = y - r_y
    result = []
    for i in range(0,3):
        result.append((1-a)*(1-b)*image[r_x,r_y,i] + a*(1-b)*image[r_x +1,r_y,i] + a*b*image[r_x+1,r_y+1,i] + (1-a)*b*image[r_x,r_y+1,i])
    return result

def warpImage(inputIm, refIm,H):
    
    #initial conversion to get the correct size to iterate over
    output = np.zeros((refIm.shape[0],refIm.shape[1],3))
    for i in range(inputIm.shape[0]):
        for j in range(inputIm.shape[1]):
            point = np.array([[i/inputIm.shape[0]*2],[j/inputIm.shape[1]*2],[1]])
            start_point = np.dot(H,point)
            x = (start_point[0]/start_point[2])/2*refIm.shape[0]
            y = (start_point[1]/start_point[2])/2*refIm.shape[1]    
            resulting_vals = inputIm[i,j,:]  
            x = int(x)
            y = int(y)
            output[x,y,0] = resulting_vals[0]
            output[x,y,1] = resulting_vals[1]
            output[x,y,2] = resulting_vals[2]
            
            
                
    output = np.asarray(output,dtype = 'uint8')          
    cv2.imshow("output", output)
    cv2.waitKey()
    
    large = np.zeros((refIm.shape[0],refIm.shape[1],3))
    for i in range(large.shape[0]):
        for j in range(large.shape[1]):
            b = sum(output[i,j,:])
            if b == 0:
                large[i,j,:] = refIm[i,j,:]
            else:
                large[i,j,:] = output[i,j,:]
                
    large = np.asarray(large,dtype='uint8')            
    cv2.imshow("merged", large)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
    
    return output, large
            