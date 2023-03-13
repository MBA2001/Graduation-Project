import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from skimage import io
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage as S
from google.cloud import storage
from google.oauth2 import service_account
from mtcnn import MTCNN
import os
import shutil

cred = credentials.Certificate("./cert.json")
firebase_admin.initialize_app(cred,{'storageBucket': 'grad-ceed6.appspot.com'})

nose_cascade = cv2.CascadeClassifier('../utils/cascade.xml')


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    credentials = service_account.Credentials.from_service_account_file("./cert.json")
    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"File {source_file_name} uploaded to {destination_blob_name}.")

def add_transparent_image(background, foreground, x_offset=None, y_offset=None):
    bg_h, bg_w, bg_channels = background.shape
    fg_h, fg_w, fg_channels = foreground.shape

    assert bg_channels == 3, f'background image should have exactly 3 channels (RGB). found:{bg_channels}'
    assert fg_channels == 4, f'foreground image should have exactly 4 channels (RGBA). found:{fg_channels}'

    # center by default
    if x_offset is None: x_offset = (bg_w - fg_w) // 2
    if y_offset is None: y_offset = (bg_h - fg_h) // 2

    w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
    h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)

    if w < 1 or h < 1: return

    # clip foreground and background images to the overlapping regions
    bg_x = max(0, x_offset)
    bg_y = max(0, y_offset)
    fg_x = max(0, x_offset * -1)
    fg_y = max(0, y_offset * -1)
    foreground = foreground[fg_y:fg_y + h, fg_x:fg_x + w]
    background_subsection = background[bg_y:bg_y + h, bg_x:bg_x + w]

    # separate alpha and color channels from the foreground image
    foreground_colors = foreground[:, :, :3]
    alpha_channel = foreground[:, :, 3] / 255  # 0-255 => 0.0-1.0

    # construct an alpha_mask that matches the image shape
    alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))

    # combine the background with the overlay image weighted by alpha
    composite = background_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask

    # overwrite the section of the background image that has been updated
    background[bg_y:bg_y + h, bg_x:bg_x + w] = composite



# #loading out data
def predict(name,imagePath):

  models = ['NoseDetection.h5','NoseDetection2.h5']
  images = []
  image = io.imread(imagePath)
  image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
  image2 = cv2.resize(image,dsize=(600,600))
  gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
  nose_rects = nose_cascade.detectMultiScale(gray,1.3,20)
  for (x,y,w,h) in nose_rects:
      # cv2.rectangle(before[i], (x,y), (x+w,y+h), (0,255,0), 3)
      break
  crop_img = image2[y:y+h, x:x+w]
    # resize the image to make them all same size to be easier to run through the CNN model in the future
  crop_img2 = crop_img
  for i in range(2):
    if i == 0:
      crop_img2 = cv2.resize(crop_img,dsize=(100,100))
      x = tf.keras.utils.img_to_array(crop_img2)
      images.append(x/255.0)
      images = np.array(images)
      # # #load model from saved file
      model = tf.keras.models.load_model(models[i])
      prd = model.predict(images)
      tf.keras.utils.array_to_img(prd[0]).save('C:\\Users\\baher\\OneDrive\\Desktop\\2dTo3D\\'+name+str(i)+'.png')
      images = []
    else:
      crop_img2 = cv2.resize(crop_img,dsize=(400,400))
      x = tf.keras.utils.img_to_array(crop_img2)
      images.append(x/255.0)
      images = np.array(images)
      # # #load model from saved file
      model = tf.keras.models.load_model(models[i])
      prd = model.predict(images)
      tf.keras.utils.array_to_img(prd[0]).save('C:\\Users\\baher\\OneDrive\\Desktop\\2dTo3D\\'+name+str(i)+'.png')
  

  # images = np.array(images)
  # # # #load model from saved file
  # model = tf.keras.models.load_model(models[modelNumber])
  # prd = model.predict(images)
  # tf.keras.utils.array_to_img(prd[0]).save('C:\\Users\\baher\\OneDrive\\Desktop\\2dTo3D\\'+name+'.png')
  # return prd[0]


# f = plt.figure()
# f.add_subplot(2,2,1)
# plt.imshow(prd[0],interpolation='nearest')
# plt.title('prediction')
# f.add_subplot(2,2,2)
# plt.imshow(images[0],interpolation='nearest')
# plt.title('image')
# plt.show()

def augment(img,name,doctor):
  image = io.imread(img)
  image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
  image2 = cv2.resize(image,dsize=(600,600))

  gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
  nose_rects = nose_cascade.detectMultiScale(gray,1.3,20)
  for (x,y,w,h) in nose_rects:
    # cv2.rectangle(before[i], (x,y), (x+w,y+h), (0,255,0), 3)
    break


  for i in range(2):
    nose =  cv2.imread('C:\\Users\\baher\\OneDrive\\Desktop\\2dTo3D\\'+name+str(i)+'.png')

    nose = cv2.resize(nose,(w,h))

    b_channel, g_channel, r_channel = cv2.split(nose)

    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 150 #creating a dummy alpha channel image.

    img_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

    x_offset = x
    y_offset = y
    background = image2
    add_transparent_image(background, img_BGRA,x_offset,y_offset)

    cv2.imwrite('C:\\Users\\baher\\OneDrive\\Desktop\\2dTo3D\\'+name+str(i)+'A.png',background)
    upload_blob(S.bucket().name,'C:\\Users\\baher\\OneDrive\\Desktop\\2dTo3D\\'+name+str(i)+'A.png',doctor+'/'+name+'/'+str(i)+'.png')


def detect_face(imagePath):
  faceCascade = cv2.CascadeClassifier('../utils/face_cascade.xml')
  #load image from url
  image = io.imread(imagePath)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
  )
  return len(faces)
   
   
def add_detections(img,name,doctor):
  image = io.imread(img)
  image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
  parent_dir = "C:/Users/baher/OneDrive/Desktop/Deep3DFaceRecon_pytorch/baher_test/"
  parent_dir2 = "C:/Users/baher/OneDrive/Desktop/Deep3DFaceRecon_pytorch/baher_test/"+doctor+'/'

  path = os.path.join(parent_dir, doctor)
  mode = 0o666
  if not os.path.exists(parent_dir2):
    os.mkdir(path, mode)

  path = os.path.join(parent_dir2, name+'_'+doctor+'_results')
  if not os.path.exists(parent_dir2+'/'+name+'_'+doctor+'_results'):
    os.mkdir(path,mode)
  
  cv2.imwrite('C:/Users/baher/OneDrive/Desktop/Deep3DFaceRecon_pytorch/baher_test/'+doctor+'/'+name+'_'+doctor+'.png',image)
  detector = MTCNN()
  detections = detector.detect_faces(image)[0]['keypoints']
  f = open("C:/Users/baher/OneDrive/Desktop/Deep3DFaceRecon_pytorch/baher_test/"+doctor+"/"+name+"_"+doctor+".txt", "a")
  f.write(str(detections['left_eye'][0])+' '+str(detections['left_eye'][1])+'\n')
  f.write(str(detections['right_eye'][0])+' '+str(detections['right_eye'][1])+'\n')
  f.write(str(detections['nose'][0])+' '+str(detections['nose'][1])+'\n')
  f.write(str(detections['mouth_left'][0])+' '+str(detections['mouth_left'][1])+'\n')
  f.write(str(detections['mouth_right'][0])+' '+str(detections['mouth_right'][1])+'\n')

  f.close()


def threeD(doctor,name):
  os.system('python C:/Users/baher/OneDrive/Desktop/Deep3DFaceRecon_pytorch/test.py --name=faceAlignment --epoch=20 --img_folder=C:/Users/baher/OneDrive/Desktop/Deep3DFaceRecon_pytorch/baher_test/'+doctor+'/')
  # os.system('C:/Users/baher/OneDrive/Desktop/GP/code/convertor.py -i C:/Users/baher/OneDrive/Desktop/Deep3DFaceRecon_pytorch/baher_test/'+doctor+'/'+name+'_'+doctor+'.obj'+' -o C:/Users/baher/OneDrive/Desktop/gradproject/assets/'+name+'_'+doctor+'.glb')
  shutil.copy('C:/Users/baher/OneDrive/Desktop/Deep3DFaceRecon_pytorch/baher_test/'+doctor+'/'+name+'_'+doctor+'.obj','C:/Users/baher/OneDrive/Desktop/gradproject/assets/'+name+'_'+doctor+'.obj')
  shutil.copy('C:/Users/baher/OneDrive/Desktop/Deep3DFaceRecon_pytorch/baher_test/'+doctor+'/'+name+'_'+doctor+'.mat','C:/Users/baher/OneDrive/Desktop/gradproject/assets/'+name+'_'+doctor+'.mat')
  


