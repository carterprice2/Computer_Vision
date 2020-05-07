# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 15:57:52 2019

@author: Carter
"""
# import the necessary packages
from sklearn.cluster import KMeans
import numpy as np
import cv2
    
#[outputImg, meanHues] = quantizeHSV(origImg, k) 

def quantizeHSV(image, k):
    
    # get image size
    (h, w) = image.shape[:2]
     
    #convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # reshape the image into a feature vector so that k-means can be applied
    hsv = hsv.reshape((image.shape[0] * image.shape[1], 3))
    
    #grab only the h channel
    h_chan = hsv[:,0]
    h_chan = h_chan.reshape(-1,1)
    
    # apply k-means using the specified number of clusters and
    # then create the quantized image based on the predictions
    clt = KMeans(n_clusters = k)
    labels = clt.fit_predict(h_chan)
    quant = clt.cluster_centers_.astype("uint8")[labels]
    quant = np.hstack([quant,hsv[:,1:]])
    
    # reshape the feature vectors to images
    quant = quant.reshape((h, w, 3))
    
    #convert the hue based version back to 
    quant = cv2.cvtColor(quant, cv2.COLOR_HSV2BGR)
    
    return quant, clt.cluster_centers_

#    # display the images and wait for a keypress
#    cv2.imshow("image", np.hstack([image, quant]))
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
#    
#    cv2.imshow("quant", quant)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
#    
#    cv2.imshow("image", image)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()