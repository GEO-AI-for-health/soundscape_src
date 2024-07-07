# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 14:26:03 2023

@author: user
"""

import matplotlib
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import h5py
from tensorflow import keras
from tensorflow.keras import  Model
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, Callback
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
import matplotlib.pyplot as plt


batch_size = 64


# MixUP
class MixupGenerator():
    def __init__(self, X_train, y_train, batch_size=64, alpha=0.2, shuffle=True, datagen=None):
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.sample_num = len(X_train)
        self.datagen = datagen

    def __call__(self):
        while True:
            indexes = self.__get_exploration_order()
            itr_num = int(len(indexes) // (self.batch_size * 2))

            for i in range(itr_num):
                batch_ids = indexes[i * self.batch_size * 2:(i + 1) * self.batch_size * 2]
                X, y = self.__data_generation(batch_ids)

                yield X, y

    def __get_exploration_order(self):
        indexes = np.arange(self.sample_num)

        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, batch_ids):
        _, h, w, c = self.X_train.shape
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)

        X1 = self.X_train[batch_ids[:self.batch_size]]
        X2 = self.X_train[batch_ids[self.batch_size:]]
        X = X1 * X_l + X2 * (1 - X_l)

        if self.datagen:
            for i in range(self.batch_size):
                X[i] = self.datagen.random_transform(X[i])
                X[i] = self.datagen.standardize(X[i])

        if isinstance(self.y_train, list):
            y = []

            for y_train_ in self.y_train:
                y1 = y_train_[batch_ids[:self.batch_size]]
                y2 = y_train_[batch_ids[self.batch_size:]]
                y.append(y1 * y_l + y2 * (1 - y_l))
        else:
            y1 = self.y_train[batch_ids[:self.batch_size]]
            y2 = self.y_train[batch_ids[self.batch_size:]]
            y = y1 * y_l + y2 * (1 - y_l)

        return X, y


def vggmodel(nb_classes=2, input_shape=(224, 224, 3)):
    input_dim = Input(input_shape)

    x = Conv2D(64, kernel_size=3, activation="relu",padding='same')(input_dim)
    x = Conv2D(64, kernel_size=3, activation="relu",padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(128, kernel_size=3, activation="relu", padding='same')(x)
    x = Conv2D(128, kernel_size=3, activation="relu", padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(256, kernel_size=3, activation="relu", padding='same')(x)
    x = Conv2D(256, kernel_size=3, activation="relu", padding='same')(x)
    # x = Conv2D(256, kernel_size=3, activation="relu", padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(512, kernel_size=3, activation="relu", padding='same')(x)
    # x = Conv2D(512, kernel_size=3, activation="relu", padding='same')(x)
    # x = Conv2D(512, kernel_size=3, activation="relu", padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # x = Conv2D(512, kernel_size=3, activation="relu", padding='same')(x)
    # x = Conv2D(512, kernel_size=3, activation="relu", padding='same')(x)
    # x = Conv2D(512, kernel_size=3, activation="relu", padding='same')(x)
    # x = MaxPooling2D(pool_size=(2, 2))(x)

    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    #
    # # model = Flatten()(model)
    x = Dense(256, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    #
    x = Dense(nb_classes, activation="softmax")(x)
    model = Model(inputs = input_dim, outputs= x)
    return model


# plot loss and acc
class SGDLearningRateTracker(Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}

        self.acc = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.acc['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))
        # self.loss_plot('batch')

    def on_epoch_end(self, epoch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.acc['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))
        self.loss_plot('epoch')

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        matplotlib.use('AGG')
        plt.figure()
        myfig = plt.gcf()
        # acc
        plt.plot(iters, self.acc[loss_type], 'r', label='training accuracy')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='training loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='validation accuracy')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='validation loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc='best')

        myfig.savefig(
            '3wavesbestmodels/q4_02_1-3_8-10vgg/' + str(
                n_fold) + '_' + 'figuresgdl.png', dpi=200)
        myfig.clf()
        plt.close('all')


# load data

with h5py.File(
        'data_h5_new/new_subcategories/q4_02_1-3_8-10_train_data_shuffled.h5') as hf:
    train_X = hf['x'][:]
    train_Y = hf['y'][:]

print("train_X.shape:", train_X.shape)
print("len(train_Y):", len(train_Y))

# 5-fold
n_splits = 5
k_fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=7)
n_fold = 0
for train_index, val_index in k_fold.split(train_X, train_Y):
    print("n_flod:{}.".format(n_fold))
    # train_data
    X_train = train_X[train_index]
    y_train = to_categorical(train_Y[train_index], num_classes=2)
    # val_data
    X_val = train_X[val_index]
    y_val = to_categorical(train_Y[val_index], num_classes=2)
    # build model
    m = vggmodel()

    optimizer = Adam(lr=0.001)
    # optimizer=SGD(lr=0.001,momentum=0.9)

    m.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
    # print(m.summary())
    # early_stop
    early_stop = EarlyStopping(monitor="val_loss", min_delta=0, patience=30, mode="auto")
    save_best_model = keras.callbacks.ModelCheckpoint(
        '3wavesbestmodels/q4_02_1-3_8-10vgg/'
        + str(n_fold) + '_'
        + 'weights.{epoch:02d}-{acc:.4f}-{val_acc:.4f}.hdf5',
        monitor='val_acc', verbose=0, save_best_only=True,
        save_weights_only=False, mode='auto', period=1)
    ReduceLROnPlateau_func = keras.callbacks.ReduceLROnPlateau(
        monitor='val_acc',
        factor=0.1,
        patience=15, min_lr=0.0000001, verbose=1)

    datagen = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)

    training_generator = MixupGenerator(X_train, y_train, batch_size=batch_size, alpha=0.3,
                                        datagen=datagen)()
    # remote_cho = callbacks.RemoteMonitor(root='http://localhost:9000')
    history = m.fit(x=training_generator,
                    steps_per_epoch=X_train.shape[0] // batch_size,
                    validation_data=(X_val, y_val),
                    epochs=150, verbose=1,
                    callbacks=[early_stop, save_best_model, ReduceLROnPlateau_func, SGDLearningRateTracker()])

    print(history.history.keys())

    n_fold += 1
