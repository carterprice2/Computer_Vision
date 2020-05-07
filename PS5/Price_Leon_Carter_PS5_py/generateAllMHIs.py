# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 19:27:29 2019

@author: Carter
"""
import numpy as np
from computeHMI import computeMHI
import matplotlib.pyplot as plt
import matplotlib.cm as cm


parent_dir = 'PS5_Data/PS5_Data/'
actions = ['botharms', 'crouch', 'leftarmup', 'punch', 'rightkick']

MHI = {}
for action in actions:
    MHI[action]= computeMHI(parent_dir + action)
    
AllMHI = []
for action in actions:
    temp = MHI[action]
    for i in range(len(temp)):
        AllMHI.append(temp[i])

AllMHI = np.asarray(AllMHI)
#AllMHI = AllMHI.reshape(AllMHI.shape[1],AllMHI.shape[2],20)
np.save('allMHIs.npy', AllMHI)


#plot the MHIs

MHIs = np.load('allMHI_3.npy')

fig = plt.figure() 
for f in range(len(MHIs)):
    im = MHIs[f,:,:]
    a = fig.add_subplot(5, 4, f+1)
    imgplot = plt.imshow(im, cmap = cm.Greys_r)
    a.set_title(str(f+1))
    
    
fig = plt.figure() 
j = 1
for f in range(0,4):
    im = MHIs[f,:,:]
    a = fig.add_subplot(1, 3, j)
    imgplot = plt.imshow(im, cmap = cm.Greys_r)
    a.set_title(str(j))
    j += 1