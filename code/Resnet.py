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
    img = tf.keras.utils.load_img('../HDA-PlasticSurgery/NOT/before/'+str(i+1)+'.jpg')
    x = tf.keras.utils.img_to_array(img)
    before_test.append(x/255.0)

    img = tf.keras.utils.load_img('../HDA-PlasticSurgery/NOT/after/'+str(i+1)+'.jpg')
    x = tf.keras.utils.img_to_array(img)
    after_test.append(x/255.0)
  else:
    img = tf.keras.utils.load_img('../HDA-PlasticSurgery/NOT/before/'+str(i+1)+'.jpg')
    x = tf.keras.utils.img_to_array(img)
    before_train.append(x/255.0)

    img = tf.keras.utils.load_img('../HDA-PlasticSurgery/NOT/after/'+str(i+1)+'.jpg')
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


# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
# model.add(tf.keras.layers.MaxPooling2D((2, 2)))
# model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(tf.keras.layers.MaxPooling2D((2, 2)))
# model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
# for _ in range(100):
#   model.add(tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'))
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(30000, activation='softmax'))
# model.add(tf.keras.layers.Reshape((100,100,3)))

# model.summary()
# model.compile(metrics=['accuracy'],loss='binary_crossentropy')

# model.fit(before_train,after_train,epochs=10)

# model.save('VGG.h5')

model = tf.keras.models.load_model('VGG.h5')
prd = model.predict(before_test)
f = plt.figure()
f.add_subplot(2,2,1)
plt.imshow(prd[0], interpolation='nearest')
plt.title("Prediction")
f.add_subplot(2,2,2)
plt.imshow(before_test[0], interpolation='nearest')
plt.title("Before surgery")
f.add_subplot(2,2,3)
plt.imshow(after_test[0], interpolation='nearest')
plt.title("after surgery")
plt.show()






