
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt


nose_cascade = cv2.CascadeClassifier('./utils/cascade.xml')




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






before = []
after = []

for i in range(174):
  # if statement is used because images 1->9 have a 0 before them
  if i < 9:
    #images of before the surgery
    image = cv2.imread('./HDA-PlasticSurgery/Nose/'+'0'+str(i+1)+'_b.jpg')
    image2 = cv2.resize(image,dsize=(600,600))
    before.append(image2)

    #Images of after the surgery
    image = cv2.imread('./HDA-PlasticSurgery/Nose/'+'0'+str(i+1)+'_a.jpg')
    image2 = cv2.resize(image,dsize=(600,600))
    after.append(image2)
  else:
    image = cv2.imread('./HDA-PlasticSurgery/Nose/'+str(i+1)+'_b.jpg')
    image2 = cv2.resize(image,dsize=(600,600))
    before.append(image2)

    image = cv2.imread('./HDA-PlasticSurgery/Nose/'+str(i+1)+'_a.jpg')
    image2 = cv2.resize(image,dsize=(600,600))
    after.append(image2)



gray = cv2.cvtColor(after[90], cv2.COLOR_BGR2GRAY)
nose_rects = nose_cascade.detectMultiScale(gray,1.3,20)
for (x,y,w,h) in nose_rects:
  # cv2.rectangle(before[i], (x,y), (x+w,y+h), (0,255,0), 3)
  break

crop_img = after[90][y:y+h, x:x+w]
# crop_img2 = cv2.resize(crop_img,dsize=(100,100))
img = cv2.detailEnhance(crop_img, sigma_s=10, sigma_r=0.15)
img = cv2.edgePreservingFilter(img, flags=1, sigma_s=64, sigma_r=0.2)

b_channel, g_channel, r_channel = cv2.split(img)

alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 50 #creating a dummy alpha channel image.

img_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

x_offset = x
y_offset = y
background = before[90]
combined = add_transparent_image(background, img_BGRA,x_offset,y_offset)

cv2.imwrite('combined90.png',background)

