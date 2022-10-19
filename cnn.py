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
    img = tf.keras.utils.load_img('./HDA-PlasticSurgery/Nose_only_test/before/'+str(i+1)+'.jpg')
    x = tf.keras.utils.img_to_array(img)
    before_test.append(x/255.0)

    img = tf.keras.utils.load_img('./HDA-PlasticSurgery/Nose_only_test/after/'+str(i+1)+'.jpg')
    x = tf.keras.utils.img_to_array(img)
    after_test.append(x/255.0)
  else:
    img = tf.keras.utils.load_img('./HDA-PlasticSurgery/Nose_only_test/before/'+str(i+1)+'.jpg')
    x = tf.keras.utils.img_to_array(img)
    before_train.append(x/255.0)

    img = tf.keras.utils.load_img('./HDA-PlasticSurgery/Nose_only_test/after/'+str(i+1)+'.jpg')
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


# conv_layers = 4
# n_filters = 64
# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Conv2D(3,(1,1),input_shape=(400, 400,3)))
# for _ in range(conv_layers):
#   model.add(tf.keras.layers.Conv2D(n_filters, (3,3), padding='same', activation='relu'))

# model.add(tf.keras.layers.Conv2D(3, (3,3), padding='same', activation='relu'))

# model.summary()
# model.compile(metrics=['accuracy'],loss='binary_crossentropy')

# model.fit(before_train,after_train,epochs=10)

# model.save('NoseDetection.h5')

# results = model.evaluate(before_test,after_test,batch_size=128)
# print("test loss, test acc:", results)


model = tf.keras.models.load_model('NoseDetection.h5')
prd = model.predict(before_test)
plot = plt.imshow(prd[0], interpolation='nearest')
plt.show()

with open('my_array.csv', 'w') as my_file:
        for i in before_test[0]:
            np.savetxt(my_file,i)
print('Array exported to file')





