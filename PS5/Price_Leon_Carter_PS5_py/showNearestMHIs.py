# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 11:09:00 2019

@author: Carter
"""

from predictAction import predictAction
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#show Nearest MHIs

def showNearestMHIs(num, hv, MHIs, labels):
    
    
#    length = len(hv)
#    num = 5
    
    test_m = []
    test_l = []
    train_m = hv
    train_l = labels
    for i in range(1):
        rand = num
        test_m.append(hv[rand,:])
        test_l.append(labels[rand,0])
        train_m = np.delete(train_m,rand, 0)
        train_l = np.delete(train_l,rand, 0)
    
    pred_labs = predictAction(test_m[0], train_m, np.ravel(train_l), debug = False, k = 4, return_all = True)
    
    print(pred_labs)
    
    plt.imshow(MHIs[num,:,:])
    plt.title("original")
    
    fig = plt.figure() 
    j = 1
    for f in pred_labs:
        print('f', f)
        im = MHIs[f,:,:]
        a = fig.add_subplot(1, 4, j)
        imgplot = plt.imshow(im, cmap = cm.Greys_r)
        a.set_title(str(j))
        j += 1
    
#hv = np.load('huVectors_3.npy')
hv = np.load('flat_ims.npy')      
MHIs = np.load('allMHI_3.npy')
labels = np.load('labels.npy')

showNearestMHIs(18,hv,MHIs,labels)