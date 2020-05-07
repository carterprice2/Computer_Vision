# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 20:11:41 2019

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
#fullframeQuieries

centroids = np.load('centroids.npy')

framesdir = 'frames/'
siftdir = 'sift/'

# Get a list of all the .mat file in that directory.
# there is one .mat file per image.
fnames = glob.glob(siftdir + '*.mat')
fnames = [i[-27:] for i in fnames]

#calculate the BOW histogram for all images 
freq = []
for i in range(len(fnames)):
    fname = siftdir + fnames[i]
    try:
        mat = scipy.io.loadmat(fname)
        idx,_ = vq(mat['descriptors'],centroids)
        c = Counter(idx)
        freq.append((fname,c))
    except:
        print('bad file', fname)
 
#this frequencey was saved as a numpy array so that it did not need to be recomputed       

#image 1
im = misc.imread(framesdir + 'friends_0000003389.jpeg')
plt.imshow(im)
misc.imsave('original.jpg', im)
freq = np.asarray(freq)

#don't forget to change this!
im_c = freq[3328,1]
im_c = dict(im_c)


#run this once image is selected above to find the closest match
#_________________________________________________

#computes the L2 norm
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

#sorts the similarities to get the top 5
import copy    
sims_copy = copy.deepcopy(sims)
sims_copy.sort()
top_5 = sims_copy[-6:]

#saves the top 5 images to file
for i in range(5):
    val = top_5[i]
    ind = sims.index(val)
    pic = freq[ind,0]
    pic = pic[5:-4]
    print(pic)
    im = misc.imread(framesdir + pic)
    misc.imsave('similar_pic_' + str(i)+ '.jpg', im)
    plt.imshow(im)
    
    
