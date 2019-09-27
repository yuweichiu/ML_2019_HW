# -*- coding: utf-8 -*-
"""

Created on : 2019/7/3
@author: Ivan Chiu
"""

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.layers import Flatten, LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.initializers import Constant, truncated_normal
from keras.optimizers import SGD, Adam
from keras_preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras import backend as K
import src.utils as utils
import src.keras_tools as knt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

############################################################
#  Session Setting
############################################################
# If you face the error about convolution layer,
# use this block to enable the memory usage of GPU growth.
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

############################################################
# Data processing
############################################################
train_data = pd.read_csv('./data/ml2019spring-hw3/train_data.csv', header=None)
train_label = pd.read_csv('./data/ml2019spring-hw3/train_label.csv', header=None)
test_data = pd.read_csv('./data/ml2019spring-hw3/test_data.csv', header=None)

# normalize and one-hot:
train_data = train_data.values/255
test_data = test_data.values/255
train_label = np_utils.to_categorical(train_label.values.flatten(), 7)

# make sure the input data shape:
train_data = np.reshape(train_data, [-1, 48, 48, 1])
test_data = np.reshape(test_data, [-1, 48, 48, 1])

############################################################
# Model
############################################################
# make around training parameters:
conv_layers = [[64, 64, 0, 128, 128, 0, 256, 256, 0, 512, 512, 0]]
kernel_size = [[3, 3, 0, 3, 3, 0, 3, 3, 0, 3, 3, 0]]
dense_layers = [[1024, 512]]
dp_layers = [0.5]
activa_fn = [['leaky_relu', 0.02], ['relu', 'relu']]
learn_rate = [0.001]
b_init = [['Constant', '0.01', Constant(0.01)]]
w_init = [['he_normal', 'he_normal', 'he_normal'], ['truncate_normal', 'M:0/S:0.02', truncated_normal(0, 0.02)]]
epochs = 100
validate_rate = 0.2
bt_size = [256]

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

# N-fold Validation:
n_splits = round(1/validate_rate)
nflist = utils.N_Fold_Validate(n_splits, train_data.shape[0])

# create project:
tn = time.localtime()
project = "./logs/hw3/D{0:4d}{1:02d}{2:02d}T{3:02d}{4:02d}".format(tn[0], tn[1], tn[2],
                                                                   tn[3], tn[4])
os.mkdir(project)

net_count = 1
for clid, cl in enumerate(conv_layers):
    ks = kernel_size[clid]
    for denl in dense_layers:
        for dp in dp_layers:
            for af in activa_fn:
                for lr in learn_rate:
                    for wi in w_init:
                        for bi in b_init:
                            for bs in bt_size:
                                dir = os.path.join(project, "network" + str(net_count))
                                os.mkdir(dir)
                                param = {}
                                param['Epochs'] = epochs
                                param['Validate_Rate'] = validate_rate
                                param['Batch_size'] = bs
                                param['Learning_Rate'] = lr
                                param['Weight_initializer'] = wi[0] + " " + wi[1]
                                param['Bias_initializer'] = bi[0] + " " + bi[1]
                                max_val_acc = {}
                                for nfid, nf in enumerate(nflist):
                                    train_id, valid_id = nf
                                    x_valid, y_valid = train_data[valid_id], train_label[valid_id]
                                    x_train, y_train = train_data[train_id], train_label[train_id]
                                    train_generator = datagen.flow(
                                        train_data[train_id], train_label[train_id],
                                        batch_size=bs,
                                    )
                                    validate_generator = datagen.flow(
                                        train_data[valid_id], train_label[valid_id],
                                        batch_size=bs,
                                    )
                                    K.clear_session()
                                    model = Sequential()
                                    for cid, cll in enumerate(cl):
                                        if cid == 0:
                                            if ks[cid] == 4:
                                                stride = 2
                                            else:
                                                stride = 1
                                            model.add(Conv2D(cll, (ks[cid], ks[cid]), strides=stride, padding='same',
                                                             kernel_initializer=wi[2],
                                                             bias_initializer=bi[2], input_shape=(48, 48, 1)))
                                            model.add(BatchNormalization())
                                            if af[0] == 'relu':
                                                model.add(Activation('relu'))
                                            elif af[0] == 'leaky_relu':
                                                model.add(LeakyReLU(af[1]))
                                        elif cll == 0:
                                            model.add(MaxPool2D((2, 2), strides=2, padding="same"))
                                        else:
                                            model.add(Conv2D(cll, (ks[cid], ks[cid]), strides=1, padding='same',
                                                             kernel_initializer=wi[2], bias_initializer=bi[2]))
                                            model.add(BatchNormalization())
                                            if af[0] == 'relu':
                                                model.add(Activation('relu'))
                                            elif af[0] == 'leaky_relu':
                                                model.add(LeakyReLU(af[1]))

                                    model.add(Flatten())
                                    for dnl in denl:
                                        model.add(Dense(units=dnl, kernel_initializer=wi[2], bias_initializer=bi[2]))
                                        model.add(BatchNormalization())
                                        if af[0] == 'relu':
                                            model.add(Activation('relu'))
                                        elif af[0] == 'leaky_relu':
                                            model.add(LeakyReLU(af[1]))

                                        model.add(Dropout(dp))

                                    model.add(Dense(units=7, kernel_initializer=wi[2]))
                                    model.add(Activation('softmax'))
                                    net_rpt = knt.model_summary(model, param)
                                    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr), metrics=['accuracy'])
                                    mc = ModelCheckpoint(os.path.join(dir, 'best_model_fold' + str(nfid+1) + '.h5'), monitor='val_acc', mode='max', verbose=1, save_best_only=True)
                                    # hist = model.fit(x_train, y_train,
                                    #                  batch_size=bs, epochs=epochs,
                                    #                  validation_data=(x_valid, y_valid), callbacks=[mc])

                                    hist = model.fit_generator(
                                        train_generator,
                                        steps_per_epoch=train_generator.n // bs,
                                        epochs=epochs,
                                        validation_data=validate_generator,
                                        validation_steps=validate_generator.n // bs,
                                        callbacks=[mc]
                                    )
                                    acc_train = hist.history['acc']
                                    loss_train = hist.history['loss']
                                    acc_valid = hist.history['val_acc']
                                    loss_valid = hist.history['val_loss']
                                    np.savetxt(os.path.join(dir, 'acc_fold' + str(nfid+1) + '.txt'), acc_train)
                                    np.savetxt(os.path.join(dir, 'loss_fold' + str(nfid+1) + '.txt'), loss_train)
                                    np.savetxt(os.path.join(dir, 'val_acc_fold' + str(nfid+1) + '.txt'), acc_valid)
                                    np.savetxt(os.path.join(dir, 'val_loss_fold' + str(nfid+1) + '.txt'), loss_valid)
                                    max_val_acc[str(nfid+1)+"-fold"] = np.max(np.asarray(acc_valid))

                                    fig = plt.figure()
                                    ax = fig.add_subplot(1, 1, 1)
                                    ax.plot(range(epochs), acc_train, label='Train')
                                    ax.plot(range(epochs), acc_valid, label='Validate')
                                    ax.legend()
                                    plt.savefig(os.path.join(dir, 'acc_fold' + str(nfid+1) + '.png'), dpi=300)
                                    plt.show(block=False)
                                    plt.close('all')

                                    knt.model_summary(model, param, max_val_acc, False, os.path.join(dir, 'summary.txt'))

                                net_count = net_count + 1
