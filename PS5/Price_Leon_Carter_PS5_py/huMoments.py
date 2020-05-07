# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 13:21:36 2019

@author: Carter
"""

import numpy as np

#try switching x and y

def moment(im,i,j):
    moment = 0
    x,y = im.shape
    for k in range(x):
        for l in range(y):
            moment = moment + k**i*l**j*im[k,l]
    return moment

def cen_mom(im,x_bar,y_bar,p,q):
    moment = 0
    x,y = im.shape
    for k in range(x):
        for l in range(y):
            moment = moment + (k-x_bar)**p*(l-y_bar)**q*im[k,l]
    return moment
    
    
def huMoments(H):
    #make sure H is normalized
    large = np.max(H) 
    H = H/large
    print(H.shape)
    
    #caluculate X_bar and Y_bar
    m00 = moment(H,0,0)
    m10 = moment(H,1,0)
    m01 = moment(H,0,1)
    
    x_bar = m10/m00
    y_bar = m01/m00
    
    #get central moments
    cm20 = cen_mom(H,x_bar,y_bar,2,0)
    cm02 = cen_mom(H,x_bar,y_bar,0,2)
    cm11 = cen_mom(H,x_bar,y_bar,1,1)
    cm30 = cen_mom(H,x_bar,y_bar,3,0)
    cm03 = cen_mom(H,x_bar,y_bar,0,3)
    cm12 = cen_mom(H,x_bar,y_bar,1,2)
    cm21 = cen_mom(H,x_bar,y_bar,2,1)
    
    #calculate Hu Moments
    h1 = cm20 + cm02
    h2 = (cm20 - cm02)**2 + 4*cm11**2
    h3 = (cm30- 3*cm12)**2 + (3*cm21-cm03)**2
    h4 = (cm30 + cm12)**2 + (cm21+cm03)**2
    h5 = (cm30 - 3*cm12)*(cm30+cm12)*((cm30+cm12)**2 -3*(cm21+cm03)**2) + (3*cm21 - cm03)*(cm21 + cm03)*(3*(cm30+cm12)**2 - (cm21 + cm03)**2)
    h6 = (cm20 - cm02)*((cm30+ cm12)**2 -(cm21 + cm03)**2) + 4*cm11*(cm30+cm12)*(cm21 + cm03)
    h7 = (3*cm21 - cm03)*(cm30+cm12)*((cm30 +cm12)**2 - 3*(cm21 + cm03)**2) - (cm30 - 3*cm12)*(cm21 + cm03)*(3*(cm30 + cm12)**2 - (cm21 +cm03)**2)
    
    
    return [h1,h2,h3,h4,h5,h6,h7]


#main code
MHIs = np.load('allMHI_2.npy')

huVectors = []
for i in range(len(MHIs)):
    im1 = MHIs[i,:,:]
    moments = huMoments(im1)
    huVectors.append(moments)


huVectors = np.asarray(huVectors)
np.save('huVectors_3.npy',huVectors)