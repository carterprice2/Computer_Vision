# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 11:49:46 2019

@author: Carter
"""

#it-idf-- term frequency and inverse docuemnt frequency
from collections import Counter
import numpy as np
import math
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

framesdir = 'frames/'
siftdir = 'sift/'

# Get a list of all the .mat file in that directory.
# there is one .mat file per image.
fnames = glob.glob(siftdir + '*.mat')
fnames = [i[-27:] for i in fnames]

freq = np.load('freq.npy', allow_pickle = True)
centroids = np.load('centroids.npy', allow_pickle = True)

# get the parameters for if-idf
Num_docs = len(freq)
big_count = {}
num_docs_in = {}

for i in range(len(centroids)):
    big_count[i] = 0
    num_docs_in[i] = 0

#get total number of words in vocabulary and get number of docs it occurs in
#big_count is the count for each word across all docs
#num in docs is the number of documents each word is in
for i in range(len(freq)):
    counter = freq[i,1]
    d = counter.items()
    for item in d:
        a = item[0]
        big_count[a] += item[1]
        num_docs_in[a] += 1

big_c = Counter(big_count) 

global big_count, num_docs_in, Num_docs
 

#use weighting for matching regions to images      
def get_idf_weight(feature, num_in_doc, total_doc):
    global big_count, num_docs_in, Num_docs
    ti = (num_in_doc/total_doc)*math.log(Num_docs/num_docs_in[feature])
    return ti


pic_id =5444
im = misc.imread(framesdir + fnames[pic_id-60][:-4])
plt.imshow(im)
#im = misc.imread(framesdir + 'friends_0000000085.jpeg')
#plt.imshow(im)
fname = siftdir + fnames[pic_id-60]
#fname = siftdir + 'friends_15170000000085.jpeg.mat'
im_mat = scipy.io.loadmat(fname)

#get the ROI
#select a region within image 1 
print ('use the mouse to draw a polygon, right click to end it')
pl.imshow(im)
MyROI = roipoly(roicolor='r')

#run this later___the blocking feature is not working in my IDE
Ind = MyROI.getIdx(im, im_mat['positions'])
print("features in region", len(Ind))

ROI_features = im_mat['descriptors'][Ind,:]

#    freq = np.load('freq.npy')
#    freq = np.asarray(freq)

idx,_ = vq(ROI_features,centroids)
c = Counter(idx)

im_c = c
im_dic = dict(im_c)

#im_t = convert_to_tf(im_dic)

def L2(lis):
    s = 0
    for item in lis:
        s = s + item**2
    return math.sqrt(s)

#converts the frequency values into the weighted tf-idf
def convert_to_tf(c):
    tot = sum(c.values())
    for key in c.keys():
        t = get_idf_weight(key,c[key],tot)
        c[key] = t
    return c

im_t = convert_to_tf(im_dic)
im_c = im_t

#remove any very common descriptors
#remove if greater than 15000
#this implements the STOP LIST
pops = []
for k in im_c.keys():
    if big_c[k] > 10000:
        pops.append(k)
for k in pops:
    im_c.pop(k) 

      
#compute similarities
norm1 = L2(im_c.values())
sims = []
for i in range(len(freq)):
    f = dict(freq[i,1])
    f = convert_to_tf(f)
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


#saves the top m images to file    
m = 10
sims_copy = copy.deepcopy(sims)
sims_copy.sort()
d = m+1
top_5 = sims_copy[-d:]
top_5.reverse()

#displays a 2x5 figure of the top 10 images
fig = plt.figure() 
for i in range(m):
    val = top_5[i]
    ind = sims.index(val)
    pic = freq[ind,0]
    pic = pic[5:-4]
    print(pic)
    im = misc.imread(framesdir + pic)
    misc.imsave('similar_pic_' + str(i)+ '.jpg', im)
    a = fig.add_subplot(2, 5, i+1)
    imgplot = plt.imshow(im)
    a.set_title(str(i+1))
    


   