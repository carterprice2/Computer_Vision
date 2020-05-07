# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 12:03:47 2019

@author: Carter
"""

import numpy as np

def computeH2(cc1,cc2):
    
    t1 = np.empty(cc1.shape)
    for i in range(len(cc1)):
        t1[i,0] = cc1[i,1] 
        t1[i,1] = cc1[i,0]

    t2 = np.empty(cc2.shape)
    for i in range(len(cc2)):
        t1[i,0] = cc2[i,1] 
        t2[i,1] = cc2[i,0]  
    
    length = len(t1)*2

    L = np.zeros((length,9))
    
    i = 0
    j = 0
    for point in t1:
       L[j,0] = t1[i,0]
       L[j,1] = t1[i,1]
       L[j,2] = 1
       L[j,6] = -t2[i,0]*t1[i,0] 
       L[j,7] = -t2[i,0]*t1[i,1]
       L[j,8] = -t2[i,0]
       
       L[j+1,3] = t1[i,0] 
       L[j+1,4] = t1[i,1]
       L[j+1,5] = 1
       L[j+1,6] = -t2[i,1]*t1[i,0] 
       L[j+1,7] = -t2[i,1]*t1[i,1]
       L[j+1,8] = -t2[i,1]
       
       i = i+1
       j = j+2
       
    print(L)
    LT = np.transpose(L)
    result = np.dot(LT,L)
    eigen = np.linalg.eig(result)
    
#    smallest_eigen_val = min(eigen[0])
    smallest_eigen_val_index = np.where(eigen[0] == min(eigen[0]))
    
    eigen_vector = eigen[1][:,smallest_eigen_val_index]
    
    h = eigen_vector.reshape((3,3))
    return h
