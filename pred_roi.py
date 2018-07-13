import os,cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
from skimage import measure
from tensorflow.python.keras.models import load_model
import scipy.io as sio
from autoROIFuncs import meanIOU


	
# Set some parameters	
model = load_model(r'D:\ML\pws_seg\pws_v3.h5',custom_objects={'mean_iou': meanIOU})
IMG_CHANNELS = 2
data = [
		(r'E:\Myokine Optimized Conditions\48 Hr 6-29-18\6-29-18 PWS 48 Hr\15uM ox Plates 7-9 24 Hr\Plate 7',(1,16)),
		(r'E:\Myokine Optimized Conditions\48 Hr 6-29-18\6-29-18 PWS 48 Hr\15uM ox Plates 7-9 24 Hr\Plate 8',(1,15)),
		(r'E:\Myokine Optimized Conditions\48 Hr 6-29-18\6-29-18 PWS 48 Hr\15uM ox Plates 7-9 24 Hr\Plate 9',(1,12)),
		(r'E:\Myokine Optimized Conditions\48 Hr 6-29-18\6-29-18 PWS 48 Hr\15uM ox Plates 7-9 24 Hr\Plate 9',(14,15)),
		
		(r'E:\Myokine Optimized Conditions\48 Hr 6-29-18\6-29-18 PWS 48 Hr\Differentiation Media 1mL Plates 1-3 24Hr\Plate 1',(1,15)),
		(r'E:\Myokine Optimized Conditions\48 Hr 6-29-18\6-29-18 PWS 48 Hr\Differentiation Media 1mL Plates 1-3 24Hr\Plate 2',(1,15)),
		(r'E:\Myokine Optimized Conditions\48 Hr 6-29-18\6-29-18 PWS 48 Hr\Differentiation Media 1mL Plates 1-3 24Hr\Plate 3',(1,15)),
		
		(r'E:\Myokine Optimized Conditions\48 Hr 6-29-18\6-29-18 PWS 48 Hr\Exercise media 0.8 mL +15uM ox Plates 10-12\Plate 10',(1,15)),
		(r'E:\Myokine Optimized Conditions\48 Hr 6-29-18\6-29-18 PWS 48 Hr\Exercise media 0.8 mL +15uM ox Plates 10-12\Plate 11',(1,15)),
		(r'E:\Myokine Optimized Conditions\48 Hr 6-29-18\6-29-18 PWS 48 Hr\Exercise media 0.8 mL +15uM ox Plates 10-12\Plate 12',(1,15)),
		
		(r'E:\Myokine Optimized Conditions\48 Hr 6-29-18\6-29-18 PWS 48 Hr\Exercise Media 0.8mL Plates 4-6 24 Hr\Plate 4',(1,15)),
		(r'E:\Myokine Optimized Conditions\48 Hr 6-29-18\6-29-18 PWS 48 Hr\Exercise Media 0.8mL Plates 4-6 24 Hr\Plate 5',(1,15)),
		(r'E:\Myokine Optimized Conditions\48 Hr 6-29-18\6-29-18 PWS 48 Hr\Exercise Media 0.8mL Plates 4-6 24 Hr\Plate 6',(1,15)),
		
		]
label='unet'
bbox_props = dict(boxstyle="circle,pad=0.3",alpha=0.5,fc="w", ec="w", lw=1)

for folder in data:
	path = folder[0]
	print (path)
	print ('cell number = %s' %(folder[1][1]-folder[1][0]+1))
	X_test = np.zeros((folder[1][1]-folder[1][0]+1, 512, 512,IMG_CHANNELS), dtype=np.uint8)
	n=0
	m=0
	for cell in range(folder[1][0],folder[1][1]+1):
		FOV =  path+'\\'+'Cell'+str(cell)
		if os.path.exists(FOV):
			if os.path.isfile(FOV+'\image_bd.tif') and  os.path.isfile(FOV+'\p0_Rms.mat'):
				# load bd_image and rms map
				image_bd = imread(FOV+'\image_bd.tif')
				image_bd =(255.0/image_bd.max())*image_bd  
				rms = sio.loadmat(FOV+'\p0_Rms.mat')
				rms = rms['cubeRms']
				rms *= 255.0/rms.max()  
				image_stack=np.stack((rms,image_bd), -1)
				image_stack= resize(image_stack, (512, 512,IMG_CHANNELS), mode='constant', preserve_range=True)
				X_test[n] = image_stack
			else: print('Either image_bd or rms map is missing.\n(Path:%s)' % FOV)
		else: print(r'Path:%s is missing' % FOV)
		n+=1

	# Prediction
	preds = model.predict(X_test, verbose=1)

	# Threshold predictions
	for cell in range(folder[1][0],folder[1][1]+1):
		y={}
		x=preds[m,:,:,0]
		m+=1
		x[np.where( x > 0.3 )]=1
		x[np.where( x !=1 )]=0
		x=x.astype('uint8')
		y['BW']=x
		name = folder[0]+'\Cell'+str(cell)+r'\BW1_combined.mat'
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
				name = folder[0]+'\Cell'+str(cell)+r'\BW'+str(BWnum)+'_'+label+'.mat'
				print (name)
				sio.savemat(name,y)
		rms = sio.loadmat(folder[0]+'\Cell'+str(cell)+'\p0_Rms.mat')
		rms = rms['cubeRms']
		rms *= 255.0/rms.max() 
		for contour in measure.find_contours(combined_mask, 0.5):
			for i in contour:
				rms[int(i[0]),int(i[1])]=np.amax(rms)
		plt.imshow(rms,cmap='gray')
		plt.title(folder[0]+'\FOV'+str(cell))
		plt.savefig(folder[0]+'\FOV'+str(cell)+'_'+label+'.png')  
		plt.close()