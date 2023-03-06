import pandas as pd
import random
import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
#loading out data
preds = []

# for i in range(57):
#   img = tf.keras.utils.load_img('../HDA-PlasticSurgery/predictions/prediction'+str(i)+'.jpg')
#   preds.append(tf.keras.utils.img_to_array(img))

for i in range(57):
  image = cv2.imread('../HDA-PlasticSurgery/predictions/prediction'+str(i)+'.jpg')
  f = plt.figure()
  f.add_subplot(2,2,1)
  xx = cv2.cvtColor(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY),cv2.COLOR_GRAY2RGB)
  cv2.imwrite('./HDA-PlasticSurgery/preds/prediction'+str(i)+'.jpg',xx)



# for i in range(10):
#   f = plt.figure()
#   f.add_subplot(2,2,1)
#   plt.imshow(prd[i], interpolation='nearest')
#   plt.title("Prediction")
#   f.add_subplot(2,2,2)
#   plt.imshow(before_test[i], interpolation='nearest')
#   plt.title("Before surgery")
#   f.add_subplot(2,2,3)
#   plt.imshow(after_test[i], interpolation='nearest')
#   plt.title("after surgery")
#   plt.show()






