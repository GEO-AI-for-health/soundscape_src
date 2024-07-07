# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 14:26:03 2023

@author: user
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Reshape,Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import to_categorical
import h5py
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import StratifiedKFold

batch_size = 64


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
        plt.plot(iters, self.acc[loss_type], 'r', label='training accuracy')
        plt.plot(iters, self.losses[loss_type], 'g', label='training loss')
        if loss_type == 'epoch':
            plt.plot(iters, self.val_acc[loss_type], 'b', label='validation accuracy')
            plt.plot(iters, self.val_loss[loss_type], 'k', label='validation loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc='best')

        myfig.savefig(r'bestmodels_2rnn/q4_02_1-3_8-10lstm/' + str(n_fold) +
            '_' + 'figuresgdl.png', dpi=200)
        myfig.clf()
        plt.close('all')



with h5py.File(
        'data_h5_new/q4_02_1-3_8-10_train_data_shuffled.h5') as hf:
    train_X = hf['x'][:]
    train_Y = hf['y'][:]

print("train_X.shape:", train_X.shape)
print("len(train_Y):", len(train_Y))

# PARAMETER
max_len = 224
lstm_units = 16
num_classes = 2
batch_size = 64
epochs = 150
# 5FOLD
n_splits = 5
k_fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=7)
n_fold = 0
for train_index, val_index in k_fold.split(train_X, train_Y):
    print("n_fold:{}.".format(n_fold))
    # TRAIN DATA
    X_train = train_X[train_index, :, :, 0]
    y_train = to_categorical(train_Y[train_index], num_classes=2)
    # VAL DATA
    X_val = train_X[val_index, :, :, 0]
    y_val = to_categorical(train_Y[val_index], num_classes=2)
    print("X_train and X_val.shape.shape:",X_train.shape,X_val.shape)
    print("y_train.shape:",y_train.shape)
    # # RESHAPE
    X_train = X_train.transpose(0, 2, 1)
    X_val = X_val.transpose(0, 2, 1)

    # BUILD MODEL
    model = Sequential()
    model.add(GRU(units=lstm_units,  return_sequences=True,activation='tanh',recurrent_activation='hard_sigmoid',
                   input_shape=(X_train.shape[2], X_train.shape[1])))
    print(X_train.shape)
    print(X_train.shape[1], X_train.shape[2])
    model.add(Dropout(0.5))
    model.add(GRU(units=lstm_units,activation='tanh',recurrent_activation='hard_sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(units=num_classes, activation='softmax'))
    model.summary()

    optimizer = Adam(lr=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc', f1_m, precision_m, recall_m])

    # early_stop
    early_stop = EarlyStopping(monitor="val_loss", patience=15, mode="auto")
    save_best_model = ModelCheckpoint('bestmodels_2rnn/q4_02_1-3_8-10GRU/'
                                      +str(n_fold) + '_'+'GRU_weights.{epoch:02d}-{acc:.4f}-{val_acc:.4f}.hdf5',
                                      monitor='val_acc',
                                      verbose=1, save_best_only=True, save_weights_only=False, mode='auto')
    ReduceLROnPlateau_func = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=5, min_lr=0.0000001, verbose=1)
    sgd_lr_tracker = SGDLearningRateTracker()
    # TRAIN MODEL
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
              validation_data=(X_val, y_val),
              callbacks=[early_stop, save_best_model, ReduceLROnPlateau_func, sgd_lr_tracker])

    n_fold += 1