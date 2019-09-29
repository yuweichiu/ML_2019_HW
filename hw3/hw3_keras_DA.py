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
import src.keras_tools as kst
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
x_train = pd.read_csv('./data/ml2019spring-hw3/train_data.csv', header=None)
y_train = pd.read_csv('./data/ml2019spring-hw3/train_label.csv', header=None)
x_test = pd.read_csv('./data/ml2019spring-hw3/test_data.csv', header=None)

# normalize and one-hot:
x_train = x_train.values/255
x_test = x_test.values/255
y_train = np_utils.to_categorical(y_train.values.flatten(), 7)

# make sure the input data shape:
x_train = np.reshape(x_train, [-1, 48, 48, 1])
x_test = np.reshape(x_test, [-1, 48, 48, 1])


############################################################
# Model
############################################################
# make around parameters:
dp_layers = 0.5
learn_rate = 0.001

# training parameters:
epochs = 100
validate_rate = 0.2
bt_size = 256

param = {}
param['Epochs'] = epochs
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

K.clear_session()
inputs = kst.md_input(shape=(48, 48, 1))
net = kst.conv2d(inputs, 64, (3, 3))
net = kst.batch_norm(net)
net = kst.activation(net, "LeakyReLU")
net = kst.conv2d(net, 64, (3, 3))
net = kst.batch_norm(net)
net = kst.activation(net, "LeakyReLU")
net = kst.maxpool2d(net)
net = kst.conv2d(net, 128, (3, 3))
net = kst.batch_norm(net)
net = kst.activation(net, "LeakyReLU")
net = kst.conv2d(net, 128, (3, 3))
net = kst.batch_norm(net)
net = kst.activation(net, "LeakyReLU")
net = kst.maxpool2d(net)
net = kst.conv2d(net, 256, (3, 3))
net = kst.batch_norm(net)
net = kst.activation(net, "LeakyReLU")
net = kst.conv2d(net, 256, (3, 3))
net = kst.batch_norm(net)
net = kst.activation(net, "LeakyReLU")
net = kst.maxpool2d(net)
net = kst.conv2d(net, 512, (3, 3))
net = kst.batch_norm(net)
net = kst.activation(net, "LeakyReLU")
net = kst.conv2d(net, 512, (3, 3))
net = kst.batch_norm(net)
net = kst.activation(net, "LeakyReLU")
net = kst.maxpool2d(net)
net = kst.flatten(net)
net = kst.dense(net, 1024)
net = kst.batch_norm(net)
net = kst.activation(net, "LeakyReLU")
net = kst.dropout(net, 0.5)
net = kst.dense(net, 512)
net = kst.batch_norm(net)
net = kst.activation(net, "LeakyReLU")
net = kst.dropout(net, 0.5)
net = kst.dense(net, 7)
net_out = kst.activation(net, "softmax")

model = Model(inputs, net_out)

net_rpt = kst.model_summary(model, param_dict=param)

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learn_rate), metrics=['accuracy'])

# create project
tn = time.localtime()
project = "./logs/hw3/D{0:4d}{1:02d}{2:02d}T{3:02d}{4:02d}".format(tn[0], tn[1], tn[2], tn[3], tn[4])
os.mkdir(project)

mc = ModelCheckpoint(os.path.join(project, 'best_model.h5'), monitor='val_acc', mode='max', verbose=1, save_best_only=True)

hist = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.n // bt_size,
    epochs=epochs,
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
ax.plot(range(epochs), acc_train, label='Train')
ax.plot(range(epochs), acc_valid, label='Validate')
ax.legend()
plt.savefig(os.path.join(project, 'training_process.png'), dpi=300)
plt.show(block=False)
plt.close('all')
