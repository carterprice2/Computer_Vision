# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 13:00:32 2019

@author: Carter
"""

#classifyAllActions

from predictAction import predictAction
import numpy as np


def predict_Leave1out(num,hv,labels):
    test_m = []
    test_l = []
    train_m = hv
    train_l = labels
    rand = num
    test_m.append(hv[rand,:])
    test_l = labels[rand,0]
    train_m = np.delete(train_m,rand, 0)
    train_l = np.delete(train_l,rand, 0)
    
    pred_lab = predictAction(test_m[0], train_m, np.ravel(train_l), k = 1)
    
    return int(pred_lab), int(test_l)


#split
#hv = np.load('huVectors_2.npy')
hv = np.load('flat_ims.npy')
labels = np.load('labels.npy')


cm = np.zeros((5,5))
right = 0
wrong = 0
for i in range(len(hv)):
    pred_lab,act = predict_Leave1out(i,hv,labels)
    if pred_lab == act:
        cm[pred_lab-1,act-1] += 1
        right += 1
    else:
        cm[pred_lab-1,act-1] += 1
        wrong += 1

print("wrong: ", wrong)
print("right: ", right)      
print("Confusion Mat: ") 
print(cm)