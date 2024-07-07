# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 14:26:03 2023

@author: user
"""
import h5py
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.utils import to_categorical
from keras.models import load_model
from sklearn.metrics import classification_report
import numpy as np
with h5py.File(
        'data_h5_new/q4_02_1-3_8-10_train_data_shuffled.h5') as hf:
    train_X = hf['x'][:]
    train_Y = hf['y'][:]

print("train_X.shape:", train_X.shape)
print("len(train_Y):", len(train_Y))
# load model
model=load_model('3wavesbestmodels/q4_02_1-3_8-10resnet'
                 '/1_weights.99-0.9248-0.9338.hdf5')

# 5fold
n_splits = 5
k_fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=7)
n_fold = 1
for train_index, val_index in k_fold.split(train_X, train_Y):
    print("n_flod:{}.".format(n_fold))
    # test data
    X_test = train_X[val_index]
    y_test = to_categorical(train_Y[val_index], num_classes=2)

    if n_fold == 1:
        y_pred = model.predict(X_test, batch_size=64, verbose=1)
        y_pred_bool = np.argmax(y_pred, axis=1)
        y_val_bool = np.argmax(y_test, axis=1)
        report = classification_report(y_val_bool, y_pred_bool, output_dict=True)
        formatted_report = '\n'.join([f'{key}: {report[key]}' for key in report])
        formatted_report = formatted_report.replace('f1-score', 'f1-score'.rjust(16)).replace('support',
                                                                                              'support'.rjust(16))
        print(formatted_report)

    else:
        pass
    n_fold += 1
