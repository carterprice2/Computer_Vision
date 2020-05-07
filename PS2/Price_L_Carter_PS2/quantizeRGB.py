# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 15:03:49 2019

@author: Carter
"""

 # import the necessary packages
from sklearn.cluster import KMeans
import numpy as np

def quantizeRGB(image, k):
    # get image width and height
    (h, w) = image.shape[:2]
     
    # reshape the image into a feature vector so that k-means
    # can be applied
    image = image.reshape((image.shape[0] * image.shape[1], 3))
     
    # apply k-means using the specified number of clusters and
    # then create the quantized image based on the predictions
    clt = KMeans(n_clusters = k)
    labels = clt.fit_predict(image)
    quant = clt.cluster_centers_.astype("uint8")[labels]
     
    meanColors = clt.cluster_centers_
    # reshape the feature vectors to images
    quant = quant.reshape((h, w, 3))
#    image = image.reshape((h, w, 3))
     
    return quant,meanColors

    # display the images and wait for a keypress
#    cv2.imshow("image", np.hstack([image, quant]))
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
#    