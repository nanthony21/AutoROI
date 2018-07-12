import os
import sys
import random
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

import tensorflow as tf

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

K.set_session
import re
import scipy.io as sio

import paths
dir = paths.training

# Set some parameters
IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 2
model_path=r'D:\ML\pws_seg\pws_v3.h5'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed

def rotate_images(X_imgs,IMG_CHANNELS):
    X_rotate = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
    k = tf.placeholder(tf.int32)
    tf_img = tf.image.rot90(X, k = k)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for img in X_imgs:
            for i in range(3):  # Rotation at 90, 180 and 270 degrees
                rotated_img = sess.run(tf_img, feed_dict = {X: img, k: i + 1})
                X_rotate.append(rotated_img)
        
    X_rotate = np.array(X_rotate, dtype = np.float32)
    return X_rotate

def flip_images(X_imgs,IMG_CHANNELS):
    X_flip = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
    tf_img1 = tf.image.flip_left_right(X)
    tf_img2 = tf.image.flip_up_down(X)
    tf_img3 = tf.image.transpose_image(X)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for img in X_imgs:
            flipped_imgs = sess.run([tf_img1, tf_img2, tf_img3], feed_dict = {X: img})
            X_flip.extend(flipped_imgs)
    X_flip = np.array(X_flip, dtype = np.float32)
    return X_flip
	
fov_count = 0
for i in dir:
	fov_count+=i[1]

# Get and resize train images and masks
X_train = np.zeros((fov_count, IMG_HEIGHT, IMG_WIDTH,IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((fov_count, IMG_HEIGHT, IMG_WIDTH,1), dtype=np.bool)
print('Getting and resizing train images and masks ... ')
sys.stdout.flush()
n=0
for path in dir:
	cellcount=range(1,path[1]+1)
	print (path[0],'cell number = ',path[1])
	for cell in cellcount:
		FOV = path[0]+'\Cell'+str(cell)
		if os.path.isdir(FOV):
			regex=re.compile(r'BW(\d+)_')
			BW_list= filter(regex.match,os.listdir(FOV))
			BW_num=[int(re.findall(regex,a)[0]) for a in BW_list]
			max_BW_num = np.max(BW_num)
			
			# load bd_image
			image_bd = imread(FOV+'\image_bd.tif')
			image_bd =(255.0/image_bd.max())*image_bd  
			rms = sio.loadmat(FOV+'\p0_Rms.mat')
			rms = rms['cubeRms']
			rms *= 255.0/rms.max()  
			image_stack=np.stack((rms,image_bd), -1)
			image_stack= resize(image_stack, (IMG_WIDTH,IMG_HEIGHT, IMG_CHANNELS), mode='constant', preserve_range=True)
			X_train[n] = image_stack
			
			mask = np.zeros((IMG_HEIGHT, IMG_WIDTH,1), dtype=np.bool)

			for BW in range(1,max_BW_num+1):
				if os.path.isfile(FOV+'\BW'+str(BW)+'_nuc.mat'):
					mask_ = sio.loadmat(FOV+'\BW'+str(BW)+'_nuc.mat')
					mask_ = mask_['BW'][:,:]	
					
					mask_=np.stack((mask_,)*1, -1)
					mask_ = resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', 
									  preserve_range=True)
					mask = np.maximum(mask, mask_)
			Y_train[n] = mask
			n+=1

X_train_rotated = rotate_images(X_train,2)
Y_train_rotated = rotate_images(Y_train,1)
X_train_flipped = flip_images(X_train,2)
Y_train_flipped = flip_images(Y_train,1)

X_train = np.concatenate((X_train,X_train_rotated,X_train_flipped), axis=0)
Y_train = np.concatenate((Y_train,Y_train_rotated,Y_train_flipped), axis=0)

# Check if training data looks all right
ix = random.randint(0, 60)
imshow(X_train[ix][:,:,1])
plt.show()
imshow(np.squeeze(Y_train[ix]))
plt.show()

# IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)
	
# U-Net model
inputs = Input((IMG_HEIGHT, IMG_WIDTH,IMG_CHANNELS))
s = Lambda(lambda x: x / 255) (inputs)

c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (s)
c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (p4)
c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (c5)

u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (u6)
c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (c6)

u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (c7)

u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (c8)

u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (c9)

outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
model.summary()

# Fit model
earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint(model_path, verbose=1, save_best_only=True)
results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=8, epochs=30, 
                    callbacks=[earlystopper, checkpointer])