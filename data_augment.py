import sys
sys.path.append("input")
import h5py
import numpy as np
import pandas as pd
from voxelgrid import VoxelGrid
from matplotlib import pyplot as plt

def Rx(angle, degrees=True):
	""" 
	"""
	if degrees:
		
		cx = np.cos(np.deg2rad(angle))
		sx = np.sin(np.deg2rad(angle))
		
	else:
		
		cx = np.cos(angle)
		sx = np.sin(angle)
		
	Rx = np.array(
	[[1  , 0  , 0  ],
	 [0  , cx , sx ],
	 [0  , -sx, cx]]
	)
	
	return Rx

def Ry(angle, degrees=True):
	
	if degrees:
		
		cy = np.cos(np.deg2rad(angle))
		sy = np.sin(np.deg2rad(angle))
		
	else:
		
		cy = np.cos(angle)
		sy = np.sin(angle)
		
	Ry = np.array(
	[[cy , 0  , -sy],
	 [0  , 1  , 0  ],
	 [sy , 0  , cy]]
	)
	
	return Ry

def Rz(angle, degrees=True):
		
	if degrees:
		
		cz = np.cos(np.deg2rad(angle))
		sz = np.sin(np.deg2rad(angle))
		
	else:
		
		cz = np.cos(angle)
		sz = np.sin(angle)
		
	Rz = np.array(
	[[cz , sz , 0],
	 [-sz, cz , 0],
	 [0  , 0  , 1]]
	)
		
	return Rz

def add_noise(xyz, strength=0.25):
	std = xyz.std(0) * strength
	noise = np.zeros_like(xyz)
	for i in range(3):
		noise[:,i] += np.random.uniform(-std[i], std[i], xyz.shape[0])
	return xyz + noise  
	
def get_augmented_data(data):
	with h5py.File("input/" + data + "_small.h5", "r") as hf:
		size = len(hf.keys())

		baseline_features = []
		convolutional_features = []
		labels = []
		for i in range(size):
			if i % 200 == 0:
				print(i, "\t processed")
				
			original_cloud = hf[str(i)]["points"][:]
			label = hf[str(i)].attrs["label"]
			
			voxelgrid = VoxelGrid(original_cloud, x_y_z=[16, 16, 16])

			baseline_features.append(voxelgrid.vector / np.max(voxelgrid.vector))
			convolutional_features.append(np.expand_dims(voxelgrid.vector.reshape(16,16,16), 3) / np.max(voxelgrid.vector))
			labels.append(label)

			s_x = np.random.normal(0, 90)
			s_y = np.random.normal(0, 90)
			s_z = np.random.normal(0, 180)

			cloud = original_cloud @ Rz(s_z) @ Ry(s_y) @ Rx(s_x)

			cloud = add_noise(cloud)

			voxelgrid = VoxelGrid(cloud, x_y_z=[16, 16, 16])

			baseline_features.append(voxelgrid.vector / np.max(voxelgrid.vector))
			convolutional_features.append(np.expand_dims(voxelgrid.vector.reshape(16,16,16), 3) / np.max(voxelgrid.vector))
			labels.append(label)
			
		print("[DONE]")

	return np.array(baseline_features), np.array(convolutional_features), np.array(labels)

X_train_baseline, X_train_convolutional, y_train = get_augmented_data("train")
X_valid_baseline, X_valid_convolutional, y_valid = get_augmented_data("valid")
X_test_baseline,  X_test_convolutional,  y_test  = get_augmented_data("test")

save = True

if save:
	with h5py.File("input/baseline_augmented_data.h5", 'w') as h5f:
		h5f.create_dataset('X_train', data=X_train_baseline)
		h5f.create_dataset('y_train', data=y_train)
		h5f.create_dataset('X_valid', data=X_valid_baseline)
		h5f.create_dataset('y_valid', data=y_valid)
		h5f.create_dataset('X_test',  data=X_test_baseline)
		h5f.create_dataset('y_test',  data=y_test)
	with h5py.File("input/convolution_augmented_data.h5", 'w') as h5f:
		h5f.create_dataset('X_train', data=X_train_convolutional)
		h5f.create_dataset('y_train', data=y_train)
		h5f.create_dataset('X_valid', data=X_valid_convolutional)
		h5f.create_dataset('y_valid', data=y_valid)
		h5f.create_dataset('X_test',  data=X_test_convolutional)
		h5f.create_dataset('y_test',  data=y_test)	



