# -*- coding: utf-8 -*-
"""
Homework 3 - Image Sentiment Classification
Created on : 2019/7/3
@author: Ivan Chiu
"""

from keras.models import Sequential, Model
from keras.optimizers import SGD, Adam
from keras_preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras import backend as K
import src.nntools as nn
import src.keras_tools as knt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# read data:
x_train = pd.read_csv('.\ml2019spring-hw3\train_data.csv', header=None)
y_train = pd.read_csv('.\ml2019spring-hw3\train_label.csv', header=None)
x_test = pd.read_csv('.\ml2019spring-hw3\test_data.csv', header=None)

# normalize and one-hot:
x_train = x_train.values/255
x_test = x_test.values/255
y_train = np_utils.to_categorical(y_train.values.flatten(), 7)

# make sure the input data shape:
x_train = np.reshape(x_train, [-1, 48, 48, 1])
x_test = np.reshape(x_test, [-1, 48, 48, 1])

# make around parameters:
dp_layers = 0.5
learn_rate = 0.001

# training parameters:
epoch = 100
validate_rate = 0.2
bt_size = 256

param = {}
param['Epoch'] = epoch
param['Validate_Rate'] = validate_rate
param['Batch_size'] = bt_size
param['Learning_Rate'] = learn_rate
param['Weight_initializer'] = 'he_normal'
param['Bias_initializer'] = 'Constant 0.01'

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=validate_rate
)
train_generator = datagen.flow(
    x_train, y_train,
    batch_size=bt_size,
    subset='training'
)
validate_generator = datagen.flow(
    x_train, y_train,
    batch_size=bt_size,
    subset='validation'
)
project = "./ckpt"
os.mkdir(project)

K.clear_session()
inputs = knt.md_input(shape=(48, 48, 1))
net = knt.conv2d(inputs, 64, (3, 3))
net = knt.batch_norm(net)
net = knt.activation(net, "LeakyReLU")
net = knt.conv2d(net, 64, (3, 3))
net = knt.batch_norm(net)
net = knt.activation(net, "LeakyReLU")
net = knt.maxpool2d(net)
net = knt.conv2d(net, 128, (3, 3))
net = knt.batch_norm(net)
net = knt.activation(net, "LeakyReLU")
net = knt.conv2d(net, 128, (3, 3))
net = knt.batch_norm(net)
net = knt.activation(net, "LeakyReLU")
net = knt.maxpool2d(net)
net = knt.conv2d(net, 256, (3, 3))
net = knt.batch_norm(net)
net = knt.activation(net, "LeakyReLU")
net = knt.conv2d(net, 256, (3, 3))
net = knt.batch_norm(net)
net = knt.activation(net, "LeakyReLU")
net = knt.maxpool2d(net)
net = knt.conv2d(net, 512, (3, 3))
net = knt.batch_norm(net)
net = knt.activation(net, "LeakyReLU")
net = knt.conv2d(net, 512, (3, 3))
net = knt.batch_norm(net)
net = knt.activation(net, "LeakyReLU")
net = knt.maxpool2d(net)
net = knt.flatten(net)
net = knt.dense(net, 1024)
net = knt.batch_norm(net)
net = knt.activation(net, "LeakyReLU")
net = knt.dropout(net, 0.5)
net = knt.dense(net, 512)
net = knt.batch_norm(net)
net = knt.activation(net, "LeakyReLU")
net = knt.dropout(net, 0.5)
net = knt.dense(net, 7)
net_out = knt.activation(net, "softmax")

model = Model(inputs, net_out)

net_rpt = nn.model_summary(model, param_dict=param)

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learn_rate), metrics=['accuracy'])
mc = ModelCheckpoint(os.path.join(project, 'best_model.h5'), monitor='val_acc', mode='max', verbose=1, save_best_only=True)

hist = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.n // bt_size,
    epochs=epoch,
    validation_data=validate_generator,
    validation_steps=validate_generator.n // bt_size,
    callbacks=[mc]
)
acc_train = hist.history['acc']
loss_train = hist.history['loss']
acc_valid = hist.history['val_acc']
loss_valid = hist.history['val_loss']
np.savetxt(os.path.join(project, 'acc.txt'), acc_train)
np.savetxt(os.path.join(project, 'loss.txt'), loss_train)
np.savetxt(os.path.join(project, 'val_acc.txt'), acc_valid)
np.savetxt(os.path.join(project, 'val_loss.txt'), loss_valid)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(range(epoch), acc_train, label='Train')
ax.plot(range(epoch), acc_valid, label='Validate')
ax.legend()
plt.savefig(os.path.join(project, 'training_process.png'), dpi=300)
plt.show(block=False)
plt.close('all')
