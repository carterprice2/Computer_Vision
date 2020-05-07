# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 16:37:35 2019

@author: Carter
"""

import numpy as np

#predictAction

labels = []
for i in range(5):
    for j in range(4):
        labels.append(i+1)

labels = np.asarray(labels). reshape((20,1))

def dist(a,b):
    if len(a) != len(b):
        print('Lengths do not match. Doofus!')
        return None
    else:
        a = np.asarray(a)
        b = np.asarray(b)
        sub = np.subtract(a,b)
        sub2 = np.power(sub,2)
        res2 = np.sum(sub2)
        res = np.sqrt(res2)
        return res
            

def predictAction(testMoments, trainMoments, trainLabels, debug = False, k = 1, return_all = False):
    """
    input: a single test moment, the trained humoments with labels
    output: a label for the test huVector
    """
    distances = []
    for j in range(len(trainMoments)):
        test = testMoments
        train = trainMoments[j]
#        if debug:
#            print('test',test)
#            print('train', train)
        dis = dist(test,train)
        distances.append(dis)
#    if debug:
#        print("distances", distances)
    sort_d = sorted(distances)
#    print(sort_d)
    best = sort_d[:k]
    ind = []
    for item in best:
        index = distances.index(item)
        ind.append(index)
    result = []
    for i in ind:
        result.append(trainLabels[i])
    if debug:
        print (best, ind)
        print ('result', result)
#        print('Distances', distances)
#            print(result)
    #find this mode in the top k results
    if return_all:
        #this will give the indices of the actual huVectors so that we can retrieve the MHIs
        fin_res = ind
    else:
        fin_res = max(set(result), key=result.count)
    return fin_res
    

#hv = np.load('huVectors_3.npy')

hv = np.load('flat_ims.npy')
length = len(hv)
#num = int(length*0.1)
num = 1 

test_m = []
test_l = []
train_m = hv
train_l = labels
for i in range(num):
    rand = 3
#    rand = np.random.randint(0,train_m.shape[0])
    test_m.append(hv[rand,:])
    test_l.append(labels[rand,0])
    train_m = np.delete(train_m,rand, 0)
    train_l = np.delete(train_l,rand, 0)

pred_labs = predictAction(test_m[0], train_m, np.ravel(train_l), debug = True, k = 1, return_all = False)