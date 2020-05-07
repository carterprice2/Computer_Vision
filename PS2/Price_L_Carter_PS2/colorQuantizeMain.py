# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 09:05:22 2019

@author: Carter
"""

# the is the main script that calls the quantization functions

import cv2
from quantizeRGB import quantizeRGB
from quantizeHSV import quantizeHSV
from computeQuantizationError import computeQuantizationError
from getHueHists import getHueHists
import matplotlib.pyplot as plt

#import the image

image = cv2.imread('fish.jpg')

#run process for a low k value
k = 3 

#run RGB quantization
quantRGB,meanColors = quantizeRGB(image,k)

#show image
cv2.imshow("image", quantRGB)
cv2.waitKey(0)
cv2.destroyAllWindows()

#save image
cv2.imwrite("quantizedRGB_k_3.jpg", quantRGB)

#run hue quantization
quantHue,meanHues = quantizeHSV(image,k)

#show image
cv2.imshow("quant Hue image", quantHue)
cv2.waitKey(0)
cv2.destroyAllWindows()

#save image
cv2.imwrite("quantizedHue_k_3.jpg", quantHue)

#compute the quantization error RGB quant
SSD_rgb = computeQuantizationError(image,quantRGB)
print(SSD_rgb)

#compute the quantization error RGB quant
SSD_hue = computeQuantizationError(image,quantHue)
print(SSD_hue)

histEqual, histClusterd = getHueHists(image,k)
print(histEqual)
print(histClusterd)
#
#
#run process for a higher k value
k = 20 

#run RGB quantization
quantRGB,meanColors = quantizeRGB(image,k)

#show image
cv2.imshow("image", quantRGB)
cv2.waitKey(0)
cv2.destroyAllWindows()

#save image
cv2.imwrite("quantizedRGB_k_" + str(k) + ".jpg", quantRGB)

#run hue quantization
quantHue,meanHues = quantizeHSV(image,k)

#show image
cv2.imshow("quant Hue image", quantHue)
cv2.waitKey(0)
cv2.destroyAllWindows()

#save image
cv2.imwrite("quantizedHue_k_" + str(k) + ".jpg", quantHue)

#compute the quantization error RGB quant
SSD_rgb = computeQuantizationError(image,quantRGB)

#compute the quantization error RGB quant
SSD_hue = computeQuantizationError(image,quantHue)

SSD_base = computeQuantizationError(image,image)

histEqual, histClusterd = getHueHists(image,k)
print(histEqual)
print(histClusterd)