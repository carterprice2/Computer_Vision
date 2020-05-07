# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 15:13:49 2019

@author: Carter
"""

#visualize vocabulary

import numpy as np
import scipy.io
import glob
from scipy import misc
import matplotlib.pyplot as plt
from displaySIFTPatches import displaySIFTPatches
from selectRegion import roipoly
from getPatchFromSIFTParameters import getPatchFromSIFTParameters
from skimage.color import rgb2gray
import matplotlib.cm as cm
import pylab as pl
from scipy.cluster.vq import kmeans,vq
from collections import Counter
#import pdb
#import cv2

# specific frame dir and siftdir
framesdir = 'frames/'
siftdir = 'sift/'

# Get a list of all the .mat file in that directory.
# there is one .mat file per image.
fnames = glob.glob(siftdir + '*.mat')
fnames = [i[-27:] for i in fnames]

a = len(fnames)-1
b = 0.1                                      #sample rate
rands = np.random.randint(0,a,int(a*b))

# load that file and stack descriptors into large data set
data = np.empty((0,128))
for i in range(len(rands)):
    fname = siftdir + fnames[rands[i]]
    mat = scipy.io.loadmat(fname)
    a = len(mat['descriptors'])
    rand_feats = np.random.randint(0,a,int(a*.1))
    data = np.vstack((data,mat['descriptors'][rand_feats,:]))

#do k-means clustering on a subset of the features
# computing K-Means
k = 1500
centroids,_ = kmeans(data,k)
# assign each sample to a cluster
idx,_ = vq(data,centroids)


#pick up here after getting centroids
centroids = np.load('centroids.npy')

#test on one picture
#im = misc.imread(framesdir + 'friends_0000000085.jpeg')
#fname = siftdir + 'friends_0000000085.jpeg.mat'
#mat = scipy.io.loadmat(fname)
#idx,_ = vq(mat['descriptors'],centroids)
#c = Counter(idx)
#plt.imshow(im)

#grab the frequencies from n images
#sub_c = Counter()
#for i in range(500):
#    sub_c = sub_c + freq[i,1]
    
"""
pic = freq[ind,0]
    pic = pic[5:-4]
    print(pic)
    im = misc.imread(framesdir + pic)
    misc.imsave('similar_pic_' + str(i)+ '.jpg', im)
    a = fig.add_subplot(2, 5, i+1)
    imgplot = plt.imshow(im)
    a.set_title(str(i+1))
 """
   

#index2 = np.where(idx == 855) 

#show the vocabular for feature 906 in 25 images

fig = plt.figure() 
f = 0
for i in range(500):
    if f > 25: 
            break
    fname = siftdir + fnames[i]
    mat = scipy.io.loadmat(fname)
    idx,_ = vq(mat['descriptors'],centroids)
    im = misc.imread(framesdir + fnames[i][:-4])
    index = np.where(idx == 906)
    for item in index[0]:
        patch_num = item
        img_patch = getPatchFromSIFTParameters(mat['positions'][patch_num,:], mat['scales'][patch_num], mat['orients'][patch_num], rgb2gray(im))
#        plt.imshow(img_patch,  cmap = cm.Greys_r)
    #    name = 'patch_7_' + str(f) + '.jpg'
    #    misc.imsave(name,img_patch)
    #    plt.show()
        f = f +1
        if f > 25: 
            break
        a = fig.add_subplot(5, 5, f)
        imgplot = plt.imshow(img_patch, cmap = cm.Greys_r)
        a.set_title(str(f))
        
    


#for each image assign the samples to a cluster
#pics_feature_assignments = []
#for name in fnames:
#    fname = siftdir + name
#    mat = scipy.io.loadmat(fname)
#    idx,_ = vq(mat['descriptors'],centroids)
#    pics_feature_assignments.append(idx)
    

     