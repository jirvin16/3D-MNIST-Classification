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

def get_augmented_data(data, voxel_dim, rotations=False):
	voxel_dims = (voxel_dim, voxel_dim, voxel_dim)
	features = []
	labels = []
	with h5py.File("input/" + data + "_small.h5", "r") as hf:
		size = len(hf.keys())
		for i in range(size):
			if i % 200 == 0:
				print(i, "\t processed")
				
			original_cloud = hf[str(i)]["points"][:]
			label = hf[str(i)].attrs["label"]
			if rotations:
				if label != 0:
					continue
			
			voxelgrid = VoxelGrid(original_cloud, x_y_z=list(voxel_dims))

			features.append(voxelgrid.vector / np.max(voxelgrid.vector))
			labels.append(label)

			if rotations:
				angles = [-60, -30, 30, 60]
				for angle in angles:
					voxelgrid = VoxelGrid(original_cloud @ Rx(angle), x_y_z=list(voxel_dims))
					features.append(voxelgrid.vector / np.max(voxelgrid.vector))
					labels.append(label)
					voxelgrid = VoxelGrid(original_cloud @ Ry(angle), x_y_z=list(voxel_dims))
					features.append(voxelgrid.vector / np.max(voxelgrid.vector))
					labels.append(label)
					voxelgrid = VoxelGrid(original_cloud @ Rz(angle), x_y_z=list(voxel_dims))
					features.append(voxelgrid.vector / np.max(voxelgrid.vector))
					labels.append(label)

				break

			if not rotations:

				s_x = np.random.normal(0, 90)
				s_y = np.random.normal(0, 90)
				s_z = np.random.normal(0, 180)

				cloud = original_cloud @ Rz(s_z) @ Ry(s_y) @ Rx(s_x)

				cloud = add_noise(cloud)

				voxelgrid = VoxelGrid(cloud, x_y_z=list(voxel_dims))

				features.append(voxelgrid.vector / np.max(voxelgrid.vector))
				labels.append(label)
			
		print("[DONE]")

	if voxel_dim == 16:
		t = 'float64'
	else:
		t = 'float32'

	return np.array(features).astype(t), np.array(labels).astype(t)

save = True

if save:
	# voxel_sizes = [8, 16, 32]
	voxel_sizes = [16]
	np.random.seed(42)
	for voxel_dim in voxel_sizes:
		X_rotations, y_rotations = get_augmented_data("train", voxel_dim, rotations=True)
		with h5py.File("data/rotations.h5", 'w') as h5f:
			h5f.create_dataset('X_train', data=X_rotations)
			h5f.create_dataset('y_train', data=y_rotations)
		with h5py.File("data/data.h5", 'w') as h5f:
			h5f.create_dataset('X_train', data=X_train)
			h5f.create_dataset('y_train', data=y_train)
			h5f.create_dataset('X_valid', data=X_valid)
			h5f.create_dataset('y_valid', data=y_valid)
			h5f.create_dataset('X_test',  data=X_test)
			h5f.create_dataset('y_test',  data=y_test)




