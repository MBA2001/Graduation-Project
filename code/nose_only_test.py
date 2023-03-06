import pandas as pd
import random
import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
#loading out data
before_train = []
before_test = []
after_train = []
after_test = []


for i in range(174):
  rand = random.randint(0,9)
  if rand == 0 or rand == 3 or rand == 5:
    img = tf.keras.utils.load_img('../HDA-PlasticSurgery/Nose_only_test/before/'+str(i+1)+'.jpg')
    x = tf.keras.utils.img_to_array(img)
    before_test.append(x/255.0)

    img = tf.keras.utils.load_img('../HDA-PlasticSurgery/Nose_only_test/after/'+str(i+1)+'.jpg')
    x = tf.keras.utils.img_to_array(img)
    after_test.append(x/255.0)
  else:
    img = tf.keras.utils.load_img('../HDA-PlasticSurgery/Nose_only_test/before/'+str(i+1)+'.jpg')
    x = tf.keras.utils.img_to_array(img)
    before_train.append(x/255.0)

    img = tf.keras.utils.load_img('../HDA-PlasticSurgery/Nose_only_test/after/'+str(i+1)+'.jpg')
    x = tf.keras.utils.img_to_array(img)
    after_train.append(x/255.0)

before_train = np.array(before_train)
before_test = np.array(before_test)
after_train = np.array(after_train)
after_test = np.array(after_test)
 
print(before_train.shape)
print(before_test.shape)
print(after_test.shape)
print(after_train.shape)


# conv_layers = 100
# n_filters = 64
# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Conv2D(3,(1,1),input_shape=(400, 400,3)))
# for _ in range(conv_layers):
#   model.add(tf.keras.layers.Conv2D(n_filters, (3,3), padding='same', activation='relu'))

# model.add(tf.keras.layers.Conv2D(3, (3,3), padding='same', activation='relu'))

# model.summary()
# model.compile(metrics=['accuracy'],loss='binary_crossentropy')

# model.fit(before_train,after_train,epochs=10)

# model.save('NoseDetection2.h5')

# results = model.evaluate(before_test,after_test,batch_size=128)
# print("test loss, test acc:", results)

# #load model from saved file
model = tf.keras.models.load_model('NoseDetection2.h5')
prd = model.predict(before_test)
for i in range(10):
  f = plt.figure()
  f.add_subplot(2,2,1)
  plt.imshow(prd[i], interpolation='nearest')
  plt.title("Prediction")
  f.add_subplot(2,2,2)
  plt.imshow(before_test[i], interpolation='nearest')
  plt.title("Before surgery")
  f.add_subplot(2,2,3)
  plt.imshow(after_test[i], interpolation='nearest')
  plt.title("after surgery")
  plt.show()






