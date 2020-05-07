# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 19:30:28 2019

@author: Carter
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

def warpImage2(inputIm, refIm,H):
    debug = False
    
    #initialize the output matrix
    output = np.zeros((1500,1200,3))
    
    #start with 4 points
#    point = np.array([[0/inputIm.shape[1]*2],[0/inputIm.shape[0]*2],[1]])
#    new_point = np.dot(H,point)
#    x = int((new_point[0]/new_point[2])/2*output.shape[1])
#    y = int(new_point[1]/new_point[2]/2*output.shape[0])
    
    #TODO indexing is messed up figure out correct indicies
    for i in range(inputIm.shape[0]):
        for j in range(inputIm.shape[1]):
            point = np.array([[i/inputIm.shape[0]*2],[j/inputIm.shape[1]*2],[1]])
            new_point = np.dot(H,point)
            x = int((new_point[0]/new_point[2])/2*refIm.shape[0])
            y = int((new_point[1]/new_point[2])/2*refIm.shape[1])
            if i == 148 and j == 160:
                print(i,j)
                print("new_point", new_point)
                print(x,y)
            resulting_vals = inputIm[i,j,:]
            x= x +500
            y = y+200
            if x > 0 and y >0:
                output[x,y,0] = resulting_vals[0]
                output[x,y,1] = resulting_vals[1]
                output[x,y,2] = resulting_vals[2]
                output = np.asarray(output,dtype = 'uint8')
                if debug:
                    print("ref point", point)
                    print("new point", new_point)
                    print("x", x)
                    print("y", y)
                    print("resulting vals", resulting_vals)
            
    cv2.imshow("output", output)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
    plt.imshow(output)
    
    return output