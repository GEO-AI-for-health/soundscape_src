# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 14:26:03 2023

@author: user
"""
import librosa
import os
import numpy as np
import pandas as pd

def qiuMFCC(wav_file_path):
    y_ps, sr = librosa.load(wav_file_path,sr=44100,mono=True) #len(y_ps=sr*duration.
    # print(len(y_ps),sr)
    # print(round(len(y_ps)/500))
    hl=len(y_ps)//223
    melspec = librosa.feature.melspectrogram(y_ps,sr, n_fft=1764,
                                             hop_length=hl,n_mels=224) #shape of output=  (n_mels, 1+ len(y_ps) // hop_length)
    mfccs = librosa.power_to_db(melspec)
    print(len(y_ps)//hl)
    if len(y_ps)//hl+1>224:
        mfccs=mfccs[:,:224]
    else:
        pass
    return mfccs

X_train=np.zeros((7051,224,224)) #sample size *n_mels*n_time_frames
Y_train=[]
xuhao=0

with open(r'label.csv') as filename:
    labeldata=pd.read_csv(filename)
    print(len(labeldata))
for root, dirs, files in os.walk(r'combined3waves_02_1-3_8-10'):
    for fn in files:
        print(xuhao)
        print(fn)
        wave_path='combined3waves_02_1-3_8-10/'+fn
        X_train[xuhao,:,:]=qiuMFCC(wave_path)
        label= labeldata.loc[xuhao,"Q4_2class"]
        print(label)
        print("---------")
        Y_train.append(label)
        xuhao += 1
print(X_train.shape)
print(Y_train)

import h5py
with h5py.File(r'sound_raw data\data_h5_new\new_subcategories\q4_02_1-3_8-10_train_data.h5', 'w') as hf:
    hf.create_dataset('x', data=X_train)
    hf.create_dataset('y', data=Y_train)