# -*- coding: utf-8 -*-
"""
Practice some fundamental usage of keras using mnist dataset.
Created on 2019/9/28 下午 11:00
@author: Ivan Y.W.Chiu
"""

from keras.datasets import mnist
from keras.datasets import cifar10
from keras.models import Sequential, Model, load_model
from keras.initializers import Constant, truncated_normal
from keras.optimizers import SGD, Adam
from keras_preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
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
#  Dataset
############################################################
dataset = 'cifar10'
if dataset == 'mnist':
    n_class = 10
    img_shape = [28, 28, 1]
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
elif dataset == 'cifar10':
    n_class = 10
    img_shape = [32, 32, 3]
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
else:
    n_class = 10
    img_shape = [28, 28, 1]
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

# # use a part of dataset:
# data_num = 50000
# x_train, y_train = x_train[:data_num], y_train[:data_num]
# x_test, y_test = x_test[:data_num], y_test[:data_num]

# Make sure the shape is N, H, W, C:
x_train = np.reshape(x_train, [-1] + img_shape)
x_test = np.reshape(x_test, [-1] + img_shape)

# Normalization:
x_train, x_test = x_train.astype('float32')/255, x_test.astype('float32')/255

# One-hot encoding:
y_train = np_utils.to_categorical(y_train, n_class).astype('float32')
y_test = np_utils.to_categorical(y_test, n_class).astype('float32')

############################################################
# Model Parameter
############################################################
# make around parameters:
dp_layers = 0.5
learn_rate = 0.001

# training parameters:
epochs = 20
validate_rate = 0.2
bt_size = 256
b_init = ['Constant', 0.01, Constant(0.01)]
w_init = ['he_normal', 'he_normal', 'he_normal']

param = {}
param['Epochs'] = epochs
param['Validate_Rate'] = validate_rate
param['Batch_size'] = bt_size
param['Learning_Rate'] = learn_rate
param['Weight_initializer'] = w_init[0] + " " + w_init[1]
param['Bias_initializer'] = b_init[0] + " " + str(b_init[1])

############################################################
# Build model:
############################################################
K.clear_session()
inputs = kst.md_input(shape=img_shape)
net = kst.conv2d(inputs, 64, (3, 3), kernel_init=w_init[2], bias_init=b_init[1])
net = kst.batch_norm(net)
net = kst.activation(net, "relu")
net = kst.maxpool2d(net)
net = kst.conv2d(net, 128, (3, 3), kernel_init=w_init[2], bias_init=b_init[1])
net = kst.batch_norm(net)
net = kst.activation(net, "relu")
net = kst.maxpool2d(net)
net = kst.conv2d(net, 256, (3, 3), kernel_init=w_init[2], bias_init=b_init[1])
net = kst.batch_norm(net)
net = kst.activation(net, "relu")
net = kst.maxpool2d(net)
net = kst.flatten(net)
net = kst.dense(net, 512, kernel_init=w_init[2], bias_init=b_init[1])
net = kst.batch_norm(net)
net = kst.activation(net, "relu")
net = kst.dropout(net, dp_layers)
net = kst.dense(net, n_class, kernel_init=w_init[2], bias_init=b_init[1])
net_out = kst.activation(net, "softmax")

model = Model(inputs, net_out)

net_rpt = kst.model_summary(model, param_dict=param)

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learn_rate), metrics=['accuracy'])

############################################################
# Callbacks:
############################################################
# create project
tn = time.localtime()
project = "./logs/practice/D{0:4d}{1:02d}{2:02d}T{3:02d}{4:02d}".format(tn[0], tn[1], tn[2], tn[3], tn[4])
os.mkdir(project)

# Prepare model model saving directory.
model_name = '%s_model_ep{epoch:03d}.h5' % "test"
filepath = os.path.join(project, model_name)

mc = ModelCheckpoint(os.path.join(project, 'best_model.h5'), monitor='val_acc', mode='max', verbose=1, save_best_only=True)


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 5, 10, 12, 18 epochs.
    Called automatically every epoch as part of callbacks during training.

    The "Epoch" shown in monitor start from 1. However, the "epoch" in
    this function will start from 0. So, to identify with the true epoch (the one shown in
    monitor), the "epoch" here should be minus 1.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 18 - 1:
        lr *= 0.5e-3
    elif epoch > 12 - 1:
        lr *= 1e-3
    elif epoch > 10 - 1:
        lr *= 1e-2
    elif epoch > 5 - 1:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


# Prepare callbacks for model saving and for learning rate adjustment.
lr_scheduler = LearningRateScheduler(lr_schedule)

checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)

tensorboard = TensorBoard(log_dir=project, histogram_freq=0, write_graph=True, write_images=False)

callbacks = [lr_scheduler, checkpoint, tensorboard]

hist = model.fit(x_train, y_train,
                 batch_size=bt_size, epochs=epochs, shuffle=True,
                 validation_split=validate_rate, callbacks=callbacks)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

acc_train = hist.history['acc']
loss_train = hist.history['loss']
np.savetxt(os.path.join(project, 'acc.txt'), acc_train)
np.savetxt(os.path.join(project, 'loss.txt'), loss_train)

acc_valid = hist.history['val_acc']
loss_valid = hist.history['val_loss']
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


#########################################################
# Load model checkpoint to resume training.
#########################################################
resume_md_path = "./logs/practice/D20190929T1257/test_model_ep012.h5"
resume_model = load_model(resume_md_path)


def get_init_epoch(resume_md_path):
    """ Get the initial epoch from given model directory
    resume_md_path (str): './path/to/model/model_name_ep{epoch:03d}.h5'
    """
    resume_md_name = os.path.basename(resume_md_path)
    init_epoch = resume_md_name.split('ep')[-1].split('.')[0]
    return int(init_epoch)


init_epoch = get_init_epoch(resume_md_path)
resume_hist = resume_model.fit(x_train, y_train,
                               batch_size=bt_size, epochs=init_epoch + epochs, shuffle=True, initial_epoch=init_epoch,
                               validation_split=0.2, callbacks=callbacks)
