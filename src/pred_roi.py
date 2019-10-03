import os,cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
from skimage import measure
from tensorflow.python.keras.models import load_model
import scipy.io as sio
from glob import glob
from autoROIFuncs import meanIOU


	
# Set some parameters	
model = load_model(r'C:\Users\Nick\Documents\Bitbucket\autoroi\pws_v3.h5',custom_objects={'mean_iou': meanIOU})
IMG_CHANNELS = 2
mainDir = r'C:\Users\Nick\Desktop\Org\Xiang segmentation'
data = [
        	mainDir + r'\4-7-18 boston new\w1'
#			(mainDir + r'\4-7-18 boston new\w2',10),
#			(mainDir + r'\4-7-18 boston new\w3',10),
#			(mainDir + r'\4-7-18 boston new\w4',10),
#			(mainDir + r'\4-7-18 boston new\w5',10),
#			(mainDir + r'\4-7-18 boston new\w6',10),
	
		]

label='unet'
bbox_props = dict(boxstyle="circle,pad=0.3",alpha=0.5,fc="w", ec="w", lw=1)

for folder in data:
    print (folder)
    paths = glob(os.path.join(folder,'Cell*\\'))
    X_test = np.zeros((len(paths), 512, 512,IMG_CHANNELS), dtype=np.uint8)
    for i, cell in enumerate(paths):
        print(cell)
        if os.path.exists(cell):
            if os.path.isfile(cell+'\image_bd.tif') and  os.path.isfile(cell+'\p0_Rms.mat'):
                # load bd_image and rms map
                image_bd = imread(cell+'\image_bd.tif')
                image_bd =(255.0/image_bd.max())*image_bd  
                rms = sio.loadmat(cell+'\p0_Rms.mat')
                rms = rms['cubeRms']
                rms *= 255.0/rms.max()  
                image_stack=np.stack((rms,image_bd), -1)
                image_stack= resize(image_stack, (512, 512,IMG_CHANNELS), mode='constant', preserve_range=True)
                X_test[i] = image_stack
            else: print('Either image_bd or rms map is missing.\n(Path:%s)' % cell)
        else: print(r'Path:%s is missing' % cell)

    # Prediction
    preds = model.predict(X_test, verbose=1)
    
    # Threshold predictions
    for i, cell in enumerate(paths):
        y={}
        x=preds[i,:,:,0]
        x[np.where( x > 0.3 )]=1
        x[np.where( x !=1 )]=0
        x=x.astype('uint8')
        y['BW']=x
        name = cell+r'\BW1_combined.mat'
        sio.savemat(name,y)
        ret, labels = cv2.connectedComponents(x.astype(np.uint8))
        BW_name = [i for i in range(1,np.amax(labels)+1)]
        combined_mask = 0
        for j in range(1,np.amax(labels)+1):
            z=np.zeros_like(labels)
            z[np.where( labels==j )]=1
            z=z.astype('uint8')
            if (np.count_nonzero(z)) > 300:
                combined_mask+=z
                M = measure.moments(z)
                cr = M[1, 0] / M[0, 0]
                cc = M[0, 1] / M[0, 0]
                BWnum=BW_name.pop(0)
                plt.annotate(str(BWnum),(cr, cc), color='black',fontsize=13,multialignment='center',bbox=bbox_props )
                y['BW']=z
                name = cell+r'\BW'+str(BWnum)+'_'+label+'.mat'
                print (name)
                sio.savemat(name,y)
        if os.path.isfile(os.path.join(cell,'p0_Rms.mat')):
            rms = sio.loadmat(os.path.join(cell,'p0_Rms.mat'))
            rms = rms['cubeRms']
            rms *= 255.0/rms.max() 
            for contour in measure.find_contours(combined_mask, 0.5):
                for i in contour:
                    rms[int(i[0]),int(i[1])]=np.amax(rms)
            plt.imshow(rms,cmap='gray')
            plt.title(cell)
            plt.savefig(os.path.join(cell,'_'+label+'.png'))
            plt.close()
        else:
            print('Could not find ', os.path.join(cell, 'p0_Rms.mat'))