# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 13:42:08 2019

@author: Carter
"""

import numpy as np

#raw image matching KNN

MHIs = np.load('allMHI_2.npy')

flat_ims = []
for i in range(len(MHIs)):
    im = MHIs[i,:,:]
    im_flat = np.ravel(im)
    flat_ims.append(im_flat)

flat_ims = np.asarray(flat_ims)

np.save('flat_ims.npy', flat_ims)