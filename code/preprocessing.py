import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

# tool for enhancing the image's quality
# https://learnopencv.com/super-resolution-in-opencv/
# sr = cv2.dnn_superres.DnnSuperResImpl_create()
 
# path = "./utils/EDSR_x4.pb"
 
# sr.readModel(path)
 
# sr.setModel("edsr",4)


# this tool is used for node detection
nose_cascade = cv2.CascadeClassifier('./utils/cascade.xml')


before = []
after = []

for i in range(174):
  # if statement is used because images 1->9 have a 0 before them
  if i < 9:
    #images of before the surgery
    image = cv2.imread('../HDA-PlasticSurgery/Nose/'+'0'+str(i+1)+'_b.jpg')
    image2 = cv2.resize(image,dsize=(600,600))
    before.append(image2)

    #Images of after the surgery
    image = cv2.imread('../HDA-PlasticSurgery/Nose/'+'0'+str(i+1)+'_a.jpg')
    image2 = cv2.resize(image,dsize=(600,600))
    after.append(image2)
  else:
    image = cv2.imread('../HDA-PlasticSurgery/Nose/'+str(i+1)+'_b.jpg')
    image2 = cv2.resize(image,dsize=(600,600))
    before.append(image2)

    image = cv2.imread('../HDA-PlasticSurgery/Nose/'+str(i+1)+'_a.jpg')
    image2 = cv2.resize(image,dsize=(600,600))
    after.append(image2)
    
for i in range(174):
  #we turn the image to grayscale for openCV to easier classify then detect the nose and create a rectangle around it
  gray = cv2.cvtColor(before[i], cv2.COLOR_BGR2GRAY)
  nose_rects = nose_cascade.detectMultiScale(gray,1.3,20)
  for (x,y,w,h) in nose_rects:
    # cv2.rectangle(before[i], (x,y), (x+w,y+h), (0,255,0), 3)
    break

  # crop the rectangle as the nose is the only feature we want in the photos
  crop_img = before[i][y:y+h, x:x+w]
  # resize the image to make them all same size to be easier to run through the CNN model in the future
  crop_img2 = cv2.resize(crop_img,dsize=(100,100))
  # using the image enhancment tools built into open CV to increase the edge detection giving the images more depth
  img = cv2.detailEnhance(crop_img2, sigma_s=10, sigma_r=0.15)
  img = cv2.edgePreservingFilter(img, flags=1, sigma_s=64, sigma_r=0.2)
  # run the enhanced image through the image enhancing tool for better resolution
  # result  = sr.upsample(img)
  #saving the image to the files
  cv2.imwrite('../HDA-PlasticSurgery/NOT/before/'+str(i+1)+'.jpg',img)
  
  # same as the one above it but for the After photos
  gray = cv2.cvtColor(after[i], cv2.COLOR_BGR2GRAY)
  nose_rects = nose_cascade.detectMultiScale(gray,1.3,20)
  for (x,y,w,h) in nose_rects:
    # cv2.rectangle(after[i], (x,y), (x+w,y+h), (0,255,0), 3)
    break
  crop_img = after[i][y:y+h, x:x+w]
  crop_img2 = cv2.resize(crop_img,dsize=(100,100))
  img = cv2.detailEnhance(crop_img2, sigma_s=10, sigma_r=0.15)
  img = cv2.edgePreservingFilter(img, flags=1, sigma_s=64, sigma_r=0.2)
  # result  = sr.upsample(img)
  cv2.imwrite('../HDA-PlasticSurgery/NOT/after/'+str(i+1)+'.jpg',img)
  