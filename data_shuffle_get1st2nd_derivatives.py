# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 14:26:03 2023

@author: user
"""
import h5py
import numpy as np
import random
with h5py.File(r'sound_raw data\data_h5_new\new_subcategories\q4_02_1-3_8-10_train_data.h5') as hf:
    train_X0 = hf['x'][:]
    train_Y = hf['y'][:]
#1st derivatives
train_X1 = np.diff(train_X0,axis=2)
mean_train_X1 = train_X1.mean(axis=2)
cloumn1_train_X1 = train_X1[:,:,0] - mean_train_X1

train_X1=np.dstack((train_X1,cloumn1_train_X1))
print('1st derivatives')
#2nd derivatives
train_X2 = np.diff(train_X1,axis=2)
mean_train_X2 = train_X2.mean(axis=2)
cloumn1_train_X2 = train_X2[:,:,0] - mean_train_X2

train_X2=np.dstack((train_X2,cloumn1_train_X2))
print('2nd derivatives')

train_X0 = train_X0[:,:,:,np.newaxis]
train_X1 = train_X1[:,:,:,np.newaxis]
train_X2 = train_X2[:,:,:,np.newaxis]
train_X = np.concatenate((train_X0,train_X1,train_X2),3)
print('concatenate channels，increase dimensions')
#shuffle
index=[i for i in range(len(train_X))]
random.shuffle(index)
train_X=train_X[index]
train_Y=train_Y[index]

print("train_X.shape:", train_X.shape)
print("len(train_Y):",len(train_Y))

with h5py.File(r'data_h5_new\new_subcategories\q4_02_1-3_8-10_train_data_shuffled.h5', 'w') as hf:
    hf.create_dataset('x', data=train_X)
    hf.create_dataset('y', data=train_Y)

