# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 18:37:16 2019

@author: Carter
"""
import cv2
import math
import numpy as np
from sklearn.cluster import KMeans
#[centers] = detectCircles(im, radius, useGradient) 

def detectCircles(image,min_rad, max_rad, useGradient):
    
    #only return centers that are over 90% of the returned
    percent = 0.7
    
    im = cv2.imread(image)
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
#    sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=5)
#    sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=5)
    blurred = cv2.GaussianBlur( gray,(5, 5), 0)
    canny_im = cv2.Canny(blurred, 50, 255)
#    kernel = np.ones((5,5), np.uint8)
#    dialated = cv2.dilate(canny_im,kernel, iterations = 3)
#    eroded = cv2.erode(dialated,kernel)
#    canny2 = cv2.Canny(eroded,100,255)
    
    cv2.imshow("canny", canny_im)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
    c_ang = [math.cos(math.radians(i)) for i in range(0,180,1)]
    s_ang = [math.sin(math.radians(i)) for i in range(0,180,1)]
    
    hough_image = canny_im.copy()
    
    H  = np.zeros((im.shape[0],im.shape[1],int((max_rad-min_rad)/2))) 
    for i in range(hough_image.shape[0]):
        for j in range(hough_image.shape[1]):
            if hough_image[i,j] > 100:
                k = 0
                for r in range(min_rad,max_rad,2):
                    for theta in range(0,180,2):
                        a = round(i - r*c_ang[theta])
                        b = round(j - r*s_ang[theta])
                        if a < im.shape[0] and b < im.shape[1]:
                            H[a,b,k] = H[a,b,k] + 1
                    k = k + 1
                        
    centers = np.where(H > np.max(H)*percent)
    
    #convert to a list of tuples
    cents = []
    for i in range(len(centers[0])):
        tup = (centers[1][i], centers[0][i],centers[2][i])
        cents.append(tup)
    
    im_copy = im.copy()
    
    x = range(min_rad,max_rad,2)
    
    for cent in cents:
         im_copy = cv2.circle(im_copy,(int(cent[0]),int(cent[1])), x[cent[2]], (0,0,255), 2)  

    #show results
    cv2.imshow("Original", im)
    cv2.waitKey()
    cv2.imshow("Final", im_copy)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
    cv2.imwrite(str(image) + "multi_radius" + "gradient_" + str(useGradient) + ".jpg",im_copy)

    return cents

#cents = detectCircles('jupiter.jpg',10,100,0)


#clustering the circles
#K_cluster = KMeans(n_clusters = 10)
#labels = K_cluster.fit_predict(cents)
#quant_circ = K_cluster.cluster_centers_.astype("uint64")[labels]
# need to somehow check for overlapping centers
           
        
#x = range(min_rad,max_rad,2)
#      
##show the cirlces
#circled = im.copy()
#for cent in cents:
#    circled = cv2.circle(circled,(cent[1],cent[0]), x[cent[2]], (255,0,0), 3)
#    
#cv2.imshow("original", im)
#cv2.waitKey()
#cv2.imshow("circled", circled)
#cv2.waitKey()
#
#cv2.imwrite("circled_jupiter.jpg", circled)
#
#cv2.destroyAllWindows()

