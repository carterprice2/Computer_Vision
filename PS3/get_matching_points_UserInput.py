# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 12:16:42 2019

@author: Carter
"""
import cv2
import os
#import numpy

def get_corresponding_points (image1_file, image2_file):
  """
  this function takes the file names of two images then it opens them both and 
  allows the user to select the matching points in both image. 
  Select the same point in both images. First in image 1 then image 2. 
  """
  #function for the mouse call back
  def place_point_1(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
          cv2.circle(img1,(x,y),3,(255,0,0),-1)
          print("Point in image 1", (x,y))
          f = open("image_1_points.txt", "a")
          f.writelines("[" + str(x) + "," + str(y)+ "]," + "\n")
          f.close()
          
  def place_point_2(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
          cv2.circle(img2,(x,y),3,(255,0,0),-1)
          print("point in image 2", (x,y))
          f = open("image_2_points.txt", "a")
          f.writelines("[" + str(x) + "," + str(y)+ "]," + "\n")
          f.close()
          
  #read both images
  img1 = cv2.imread(image1_file)
  print(img1.shape)
  img2 = cv2.imread(image2_file)
  print(img2.shape)

  #track the mouse click locations
  #openCV has a funtion for this but I don't remeber...
  #matplotlib also has a function --> matplotlib.widgets.Cursor 
#  t1 = []
#  t2 = []
  cv2.namedWindow("image 1")
  cv2.setMouseCallback("image 1", place_point_1)
  cv2.namedWindow("image 2")
  cv2.setMouseCallback("image 2", place_point_2)

  #display the images 
  while True:
      cv2.imshow("image 1", img1)
      if cv2.waitKey(20) & 0xFF == 27:
        break
      cv2.imshow("image 2", img2)
      if cv2.waitKey(20) & 0xFF == 27:
        break
  
  cv2.destroyAllWindows()