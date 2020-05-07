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

def warpImage(inputIm,H,t2):
    
    #initial conversion to get the correct size to iterate over
    max_x = int(t2[2,1])
    max_y = int(t2[2,0])
    print(max_x, max_y)
#    bounding_points = [[0,0],[inputIm.shape[0],0],[0,inputIm.shape[1]],[inputIm.shape[0],inputIm.shape[1]]]
#    for point in bounding_points:
#        point = np.array([[point[0]/inputIm.shape[0]*2],[point[1]/inputIm.shape[1]*2],[1]])
#        start_point = np.dot(H,point)
#        x = (start_point[0]/start_point[2])/2*refIm.shape[0]
#        y = (start_point[1]/start_point[2])/2*refIm.shape[1]
#        if x > max_x:
#            max_x = int(x)
#        if y > max_y:
#            max_y = int(y)
#    print("maxes", max_x, max_y)      
    
    #initialize the output matrix
    output = np.zeros((max_x,max_y,3))
    H_inv = np.linalg.inv(H)
    
#    failed = []
    #TODO indexing is messed up figure out correct indicies
#    print('x len', output.shape[0])
#    print('y len', output.shape[1])
    for i in range(int(max_x)):
        for j in range(int(max_y)):
            point = np.array([[i/max_x*2],[j/max_y*2],[1]])
            inv_point = np.dot(H_inv,point)
            x = (inv_point[0]/inv_point[2])/2*inputIm.shape[0]
            y = (inv_point[1]/inv_point[2])/2*inputIm.shape[1]
#            print(x,y)
            if x > 0 and y >0:
                try:
                    resulting_vals = bilinear_interp(x,y,inputIm)
#                    resulting_vals = inputIm[x,y,:]
                    output[i,j,0] = resulting_vals[0]
                    output[i,j,1] = resulting_vals[1]
                    output[i,j,2] = resulting_vals[2]
                except:
#                    failed.append((x,y))
                    pass
                
    output = np.asarray(output,dtype = 'uint8')          
    cv2.imshow("output", output)
    cv2.waitKey()
    
#    large = np.zeros((max(refIm.shape[0],output.shape[0]),max(refIm.shape[1],output.shape[1]),3))
#    mod_ref = cv2.copyMakeBorder(refIm,0,large.shape[0]-refIm.shape[0],0,large.shape[1]-refIm.shape[1],cv2.BORDER_CONSTANT)
#    toggle = 0
#    for i in range(large.shape[0]):
#        for j in range(large.shape[1]):
#            a = sum(mod_ref[i,j,:])
#            b = sum(output[i,j,:])
#            if b == 0:
#                large[i,j,:] = mod_ref[i,j,:]
#            elif a == 0:
#                large[i,j,:] = output[i,j,:]
#            else:
#                if toggle == 0:
#                    large[i,j,:] = output[i,j,:]
#                else:
#                    large[i,j,:] = mod_ref[i,j,:]
#                toggle = not(toggle)
#                
#    large = np.asarray(large,dtype='uint8')            
#    cv2.imshow("merged", large)
#    cv2.waitKey()
    cv2.destroyAllWindows()
    
    return output
            