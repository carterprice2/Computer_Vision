# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 18:37:16 2019

@author: Carter
"""
import cv2
import math
import numpy as np
from scipy import stats

from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
#[centers] = detectCircles(im, radius, useGradient) 


def show_im(image):
    cv2.imshow("image", image)
    cv2.waitKey()
    cv2.destroyAllWindows()

def detectCircles(image, radius, useGradient):
    
    #only return centers that are over 90% of the returned
    percent = 0.75
    r = radius
    
    im = cv2.imread(image)
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.GaussianBlur( gray,(11, 11), 0)
#    sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=5)
#    sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=5)

#    canny_im1 = cv2.Canny(blurred, 100, 200,False)
    canny_im2 = cv2.Canny(blurred, 50, 200,True)
#    kernel = np.ones((5,5), np.uint8)
#    dilated = cv2.dilate(canny_im,kernel,iterations=1)
    if useGradient:
        sobelx = cv2.Sobel(canny_im2,cv2.CV_64F,1,0,ksize=3)
        sobely = cv2.Sobel(canny_im2,cv2.CV_64F,0,1,ksize=3)
        gradient_dir =np.arctan2(sobelx,sobely)
    
#    grad = np.gradient(canny_im2)
#    print(grad, type(grad))
    
#    cv2.imshow("gradient sobel", gradient_dir)
#    cv2.waitKey()
#    cv2.destroyAllWindows()
    
#    cv2.imshow("gradient", grad)
#    cv2.waitKey()
#    cv2.destroyAllWindows()
#    
#    sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
#    sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
#    grad_mag = abs(sobelx) + abs(sobely)
    
#    cv2.imshow("canny", canny_im1)
#    cv2.waitKey()
#    cv2.imshow("canny2", canny_im2)
#    cv2.waitKey()
#    cv2.destroyAllWindows()
    
    if useGradient:
        H  = np.zeros((im.shape[0],im.shape[1])) 
        for i in range(canny_im2.shape[0]):
            for j in range(canny_im2.shape[1]):
 
                if canny_im2[i,j] > 100:
                        a = int(round(i + r*math.cos(gradient_dir[i,j])))
                        b = int(round(j + r*math.sin(gradient_dir[i,j])))
                        if a < im.shape[0] and b < im.shape[1]:
                            H[a,b] = H[a,b] + 1   
                if canny_im2[i,j] > 100:
                        a = int(round(i - r*math.cos(gradient_dir[i,j])))
                        b = int(round(j - r*math.sin(gradient_dir[i,j])))
                        if a < im.shape[0] and b < im.shape[1]:
                            H[a,b] = H[a,b] + 1  
                if canny_im2[i,j] > 100:
                        a = int(round(i + r*math.cos(gradient_dir[i,j])))
                        b = int(round(j - r*math.sin(gradient_dir[i,j])))
                        if a < im.shape[0] and b < im.shape[1]:
                            H[a,b] = H[a,b] + 1  
                if canny_im2[i,j] > 100:
                        a = int(round(i - r*math.cos(gradient_dir[i,j])))
                        b = int(round(j + r*math.sin(gradient_dir[i,j])))
                        if a < im.shape[0] and b < im.shape[1]:
                            H[a,b] = H[a,b] + 1          
    else:
        H  = np.zeros((im.shape[0],im.shape[1])) 
        for i in range(canny_im2.shape[0]):
            for j in range(canny_im2.shape[1]):
                if canny_im2[i,j] > 100:
                    for theta in range(0,180,2):
                        a = round(i - r*math.cos(math.radians(theta)))
                        b = round(j - r*math.sin(math.radians(theta)))
                        if a < im.shape[0] and b < im.shape[1]:
                            H[a,b] = H[a,b] + 1 
                        
    centers = np.where(H > np.max(H)*percent)
    
    cv2.imshow("Accumulator Array", H)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
    print(np.max(H))
    cv2.imwrite("accumulator_array" + str(image[:-4]) + str(r)+ "gradient_" + str(useGradient) + ".jpg", H)
    
    #convert to a list of tuples
    cents = []
    for i in range(len(centers[0])):
        tup = (centers[1][i], centers[0][i])
        cents.append(tup)
    
#    copy = canny_im2.copy()     
#    for cent in cents:
#         copy = cv2.circle(copy,(int(cent[0]),int(cent[1])), 2, (255,0,255), -1)
#         
#    cv2.imshow("centers", copy)
#    cv2.waitKey()
#    cv2.destroyAllWindows()
    
    ms = MeanShift(cluster_all = True)
    ms.fit(cents)
    cluster_centers= ms.cluster_centers_  
    
#    print(cluster_centers)
#    K_clut = KMeans(n_clusters = 8)
#    labels = K_clut.fit_predict(flat_H)
#    quant_circ = K_clut.cluster_centers_.astype("uint8")[labels]
#    cluster_centers = quant_circ
    
#    cv2.imshow("K-means applied", new_new_quant)
#    cv2.waitKey()
#    cv2.destroyAllWindows()
    
    
    im_copy = im.copy()
        
    for cent in cluster_centers:
         im_copy = cv2.circle(im_copy,(int(cent[0]),int(cent[1])), r, (0,0,255), 2)  

#    for cent in cents:
#         im_copy = cv2.circle(im_copy,(int(cent[0]),int(cent[1])), r, (0,0,255), 2)
    
    #show results
    cv2.imshow("Original", im)
    cv2.waitKey()
    cv2.imshow("Final", im_copy)
    cv2.waitKey()
    cv2.destroyAllWindows()
#    cv2.imwrite(str(image[:-4]) + str(r)+ "gradient_" + str(useGradient) + ".jpg",im_copy)
    return cents
  
#centers = detectCircles('jupiter.jpg', 30,0)
   
##clustering the circles
#K_clut = KMeans(n_clusters = 4)
#labels = K_clut.fit_predict(cents)
#quant_circ = K_clut.cluster_centers_.astype("uint64")[labels]
## need to somehow check for overlapping centers      
#
#ms = MeanShift()
#ms.fit(cents)
#cluster_centers= ms.cluster_centers_
#
#mode = stats.mode(quant_circ) 
#x = mode[0][0][0]
#y = mode[0][0][1]
#
#canny_im2 = cv2.circle(canny_im2,(y,x), r, (255,0,0), -1)
#canny_im2 = cv2.circle(canny_im2,(75,234), 20, (255,0,0), 3)
#canny_im2 = cv2.circle(canny_im2,(217,321), 44, (255,0,0), 3)
#canny_im2 = cv2.circle(canny_im2,(106,458), 63, (255,0,0), 3)
#     
##show the cirlces
#for cent in centers:
#    canny_im2 = cv2.circle(canny_im2,(int(cent[0]),int(cent[1])), r, (255,0,0), 3)
#    
