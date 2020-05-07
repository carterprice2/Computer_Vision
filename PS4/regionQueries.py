# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 13:32:20 2019

@author: Carter
"""

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
import math

#region quieries
#load the centroids and frequency counters for each image
centroids = np.load('centroids.npy')
freq = np.load('freq.npy')
freq = np.asarray(freq)

framesdir = 'frames/'
siftdir = 'sift/'

# Get a list of all the .mat file in that directory.
# there is one .mat file per image.
fnames = glob.glob(siftdir + '*.mat')
fnames = [i[-27:] for i in fnames]

#determine which picture to select an ROI for
pic_id = 5444
im = misc.imread(framesdir + fnames[pic_id-60][:-4])
plt.imshow(im)

fname = siftdir + fnames[pic_id-60]
im_mat = scipy.io.loadmat(fname)

#get the ROI
#select a region within image 1 
print ('use the mouse to draw a polygon, right click to end it')
pl.imshow(im)
MyROI = roipoly(roicolor='r')

#______________________________________________________________________________________
#run this later___the blocking feature is not working in my IDE
Ind = MyROI.getIdx(im, im_mat['positions'])
print("features in region", len(Ind))

ROI_features = im_mat['descriptors'][Ind,:]

#get the matches for the ROI_features to the vocabulary - centroids
idx,_ = vq(ROI_features,centroids)
c = Counter(idx)

im_c = c

def L2(lis):
    s = 0
    for item in lis:
        s = s + item**2
    return math.sqrt(s)

#compute similarities
norm1 = L2(im_c.values())
sims = []
for i in range(len(freq)):
    f = dict(freq[i,1])
    
    total = 0
    for key in im_c.keys():
        try:
            val = im_c[key] * f[key]
        except:
            val = 0
        total = total + val
    norm = L2(f.values())
    if norm ==0:
        sim = 0
    else:
        sim = 1/(norm1*norm)*total
    sims.append(sim)

import copy    
m = 10
sims_copy = copy.deepcopy(sims)
sims_copy.sort()
d = m+1
top_5 = sims_copy[-d:]


for i in range(m):
    val = top_5[i]
    ind = sims.index(val)
    pic = freq[ind,0]
    pic = pic[5:-4]
    print(pic)
    im = misc.imread(framesdir + pic)
    misc.imsave('similar_pic_' + str(i)+ '.jpg', im)
    plt.imshow(im)


