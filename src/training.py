''' USER PARAMETERS'''
IMG_CHANNELS = 2
model_path=r'C:\Users\Nick\Desktop\Org\pws_v4.h5'
seed = 3
mainDir = r'C:\Users\Nick\Desktop\Org\Xiang segmentation'
'''****************'''

import os, time, random, warnings
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
import tensorflow.python.keras.models as tfModels
from tensorflow.python.keras.layers import Input, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate
import tensorflow.python.keras.callbacks as tfCallbacks
from glob import glob
from autoROIFuncs import meanIOU
from pwspy.dataTypes import AcqDir, ImCube, Roi


warnings.filterwarnings('ignore', category=UserWarning, module='skimage')   #Filter out unwanted warnings.

random.seed(seed)   #Set the seed so we can reproduce our results.

acqs = [AcqDir(path) for path in glob(os.path.join(mainDir, '**', 'Cell*'), recursive=True)]
height, width, _ = ImCube.fromMetadata(acqs[0].pws).data.shape

# Get and resize train images and masks
X_train = np.zeros((len(acqs), height, width, 2), dtype=np.uint8)
Y_train = np.zeros((len(acqs), height, width, 1), dtype=np.bool)
print('Getting and resizing train images and masks ... ')


for i, acq in enumerate(acqs):
		# load bd_image
		image_bd = acq.pws.getThumbnail()
		rms = acq.pws.loadAnalysis('p0').rms
		rms *= 255.0/rms.max()
		image_stack = np.stack((rms, image_bd), -1)
		X_train[i, :, :, :] = image_stack

		mask = None
		for roiName, roiNum, fileFormat in acq.getRois():
			if roiName == 'nuc':
				roi = acq.loadRoi(roiName, roiNum, fileFormat)
				mask = roi.mask if mask is None else np.logical_or(mask, roi.mask)
		Y_train[i, :, :, :] = mask

'''We want our trained model to be unaffected by image rotation.
In order to expand our training dataset we rotate and flip each image and add the modified images to our dataset.'''
X_train_rotated = [np.rot90(im, rot) for im in X_train for rot in [1, 2, 3]]
Y_train_rotated = [np.rot90(im, rot) for im in Y_train for rot in [1, 2, 3]]
X_train_flipped = [np.flip(im, ax) for im in X_train for ax in [0, 1]] + [np.transpose(im, axes=(1, 0, 2)) for im in X_train] #Flip vertically, horizontally, and transpose.
Y_train_flipped = [np.flip(im, ax) for im in Y_train for ax in [0, 1]] + [np.transpose(im, axes=(1, 0, 2)) for im in Y_train]

X_train = np.concatenate((X_train,X_train_rotated,X_train_flipped), axis=0)
Y_train = np.concatenate((Y_train,Y_train_rotated,Y_train_flipped), axis=0)

# Check if training data looks all right
ix = random.randint(0, 60)
imshow(X_train[ix][:,:,1])
plt.show()
imshow(np.squeeze(Y_train[ix]))
plt.show()

	
# U-Net model
inputs = Input((height, width, 2))
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

outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)  #This final layer condenses the tensor down to only have 1 channel since we only have one output class.

model = tfModels.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[meanIOU])
model.summary()

# Fit model
earlystopper = tfCallbacks.EarlyStopping(patience=5, verbose=1)
checkpointer = tfCallbacks.ModelCheckpoint(model_path, verbose=1, save_best_only=True)
startTime = time.time()
results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=8, epochs=30, 
                    callbacks=[earlystopper, checkpointer])
print("Completed in {} seconds.".format(time.time()-startTime))