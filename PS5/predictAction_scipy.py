# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 16:37:35 2019

@author: Carter
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

#predictAction

labels = []
for i in range(5):
    for j in range(4):
        labels.append(i+1)

labels = np.asarray(labels). reshape((20,1))
np.save('labels.npy', labels)


def predictAction(testMoments, trainMoments, trainLabels):
    k = 1
    neigh = KNeighborsClassifier(k)
    neigh.fit(trainMoments,trainLabels)
    out = neigh.predict(testMoments)
    return out


#split
hv = np.load('huVectors.npy')

length = len(hv)
num = int(length*0.1)

test_m = []
test_l = []
train_m = hv
train_l = labels
for i in range(num):
    rand = np.random.randint(0,train_m.shape[0])
    test_m.append(hv[rand,:])
    test_l.append(labels[rand,0])
    train_m = np.delete(train_m,rand, 0)
    train_l = np.delete(train_l,rand, 0)

pred_labs = predictAction(test_m, train_m, np.ravel(train_l))  

cm = np.zeros((5,5))
wrong = 0
right = 0
for i in range(len(pred_labs)):
    if pred_labs[i] == test_l[i]:
        cm[pred_labs[i]-1,pred_labs[i]-1] += 1
        right += 1
    else:
        cm[pred_labs[i]-1,test_l[i]-1] += 1
        wrong += 1

print("wrong: ", wrong)
print("right: ", right)      
        
