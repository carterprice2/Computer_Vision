# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 16:59:47 2019

@author: Carter
"""

import numpy as np
#function [error] = computeQuantizationError(origImg,quantizedImg) 

def computeQuantizationError(origImg, quantImg):
    
    SSD_error = np.sum(pow((origImg-quantImg),2))
    return SSD_error