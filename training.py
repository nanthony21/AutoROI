''' USER PARAMETERS'''
IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 2
model_path=r'C:\Users\Nick\Desktop\Org\pws_v4.h5'
seed = 3
mainDir = r'C:\Users\Nick\Desktop\Org\Xiang segmentation'
trainingDirectory = [		
			mainDir + r'\4-7-18 boston new\w1',
			mainDir + r'\4-7-18 boston new\w2',
			mainDir + r'\4-7-18 boston new\w3',
			mainDir + r'\4-7-18 boston new\w4',
			mainDir + r'\4-7-18 boston new\w5',
			mainDir + r'\4-7-18 boston new\w6',
				
			mainDir + r'\4-11-18\w1',
			mainDir + r'\4-11-18\w2',
			mainDir + r'\4-11-18\w3',
			mainDir + r'\4-11-18\w4',
			mainDir + r'\4-11-18\w5',
			mainDir + r'\4-11-18\w6',
			
			mainDir + r'\4-19-18\w1',
			mainDir + r'\4-19-18\w2',
			mainDir + r'\4-19-18\w3',
			mainDir + r'\4-19-18\w4',
			mainDir + r'\4-19-18\w5',
			mainDir + r'\4-19-18\w6',
            ]
'''****************'''

import os, time, random, warnings
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.transform import resize
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import re
import scipy.io as sio
from glob import glob
from autoROIFuncs import meanIOU


warnings.filterwarnings('ignore', category=UserWarning, module='skimage')   #Filter out unwanted warnings.

random.seed(seed)   #Set the seed so we can reproduce our results.

def rotate_images(X_imgs,IMG_CHANNELS):
    '''Given an array of images this function will return an array of images that have been rotated at 90,180, and 270 degrees.'''
    assert isinstance(X_imgs,np.ndarray)
    assert isinstance(IMG_CHANNELS,int)
    X_rotate = []   # list to hold each of the rotated images.
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
    k = tf.placeholder(tf.int32)
    tf_img = tf.image.rot90(X, k = k)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for img in X_imgs:  #Loop through each image.
            for i in range(3):  # Rotation at 90, 180 and 270 degrees
                rotated_img = sess.run(tf_img, feed_dict = {X: img, k: i + 1})
                X_rotate.append(rotated_img)     
    X_rotate = np.array(X_rotate, dtype = np.float32) #convert to numpy array and return.
    return X_rotate

def flip_images(X_imgs,IMG_CHANNELS):
    '''Given a list of images this function will return an array of image that
    have been flipped vertically,horizontally, and diagonally.
    The array should be of shape (n,Height,Width,Channels) where n is the number of images.'''
    assert isinstance(X_imgs,np.ndarray)
    assert isinstance(IMG_CHANNELS,int)
    X_flip = [] #A list to store the flipped images.
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
	
fov_count = sum([i[1] for i in trainingDirectory])

# Get and resize train images and masks
X_train = np.zeros((fov_count, IMG_HEIGHT, IMG_WIDTH,IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((fov_count, IMG_HEIGHT, IMG_WIDTH,1), dtype=np.bool)
print('Getting and resizing train images and masks ... ')

for n, path in enumerate(trainingDirectory):
	paths = glob(os.path.join(path,'Cell*\\')) #Get the paths to each cell folder
	for folder in paths:
		if os.path.isdir(folder):
			regex=re.compile(r'BW(\d+)_')
			BW_list= filter(regex.match,os.listdir(folder))
			BW_num=[int(re.findall(regex,a)[0]) for a in BW_list]
			max_BW_num = np.max(BW_num)
			
			# load bd_image
			image_bd = imread(folder + '\image_bd.tif')
			image_bd =(255.0/image_bd.max())*image_bd  
			rms = sio.loadmat(folder + '\p0_Rms.mat')
			rms = rms['cubeRms']
			rms *= 255.0/rms.max()  
			image_stack=np.stack((rms,image_bd), -1)
			image_stack= resize(image_stack, (IMG_WIDTH,IMG_HEIGHT, IMG_CHANNELS), mode='constant', preserve_range=True)
			X_train[n] = image_stack
			
			mask = np.zeros((IMG_HEIGHT, IMG_WIDTH,1), dtype=np.bool)

			for BW in range(1,max_BW_num+1):
				if os.path.isfile(folder+'\BW'+str(BW)+'_nuc.mat'):
					mask_ = sio.loadmat(folder+'\BW'+str(BW)+'_nuc.mat')
					mask_ = mask_['BW'][:,:]	
					
					mask_=np.stack((mask_,)*1, -1)
					mask_ = resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', 
									  preserve_range=True)
					mask = np.maximum(mask, mask_)
			Y_train[n] = mask
			n+=1

'''We want our trained model to be unaffected by image rotation.
In order to expand our training dataset we rotate and flip each image and add the modified images to our dataset.'''
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

	
# U-Net model
inputs = Input((IMG_HEIGHT, IMG_WIDTH,IMG_CHANNELS))
s = Lambda(lambda x: x / 255) (inputs)  #Normalize 8-bit values to 1.

'''
Right now we are using 4 times fewer convolution kernels than shown in the paper.
We are also using a different pooling strategy.
'''

'''Contracting part of the U'''
c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (s)
c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)  #Downsample by a factor of 2.

c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
'''Bottom of the U'''
c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (p4)
c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (c5)
'''Expanding part of the U'''
u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c5) #This is basically an up-sampling convolution
u6 = concatenate([u6, c4])  #We then concat with the last tensor from the same level of the U in order to get our new tensor.
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

outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)  #This final layer condensers the tensor down to only have 1 channel since we only have one output class.

model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[meanIOU])
model.summary()

# Fit model
earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint(model_path, verbose=1, save_best_only=True)
startTime = time.time()
results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=8, epochs=30, 
                    callbacks=[earlystopper, checkpointer])
print("Completed in {} seconds.".format(time.time()-startTime))