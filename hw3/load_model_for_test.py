# -*- coding: utf-8 -*-
"""

Created on : 2019/7/3
@author: Ivan Chiu
"""

# from tensorflow.examples.tutorials.mnist import input_data
# import tensorflow as tf
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
from keras.datasets import mnist
from keras import backend as K
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from keras.models import load_model

# # read data:
# train_data = pd.read_csv(r'D:\YuWei\NTU\ML_HungYiLee\ml2019spring-hw3\train_data.csv', header=None)
# train_label = pd.read_csv(r'D:\YuWei\NTU\ML_HungYiLee\ml2019spring-hw3\train_label.csv', header=None)
# test_data = pd.read_csv('./data/ml2019spring-hw3/test_data.csv', header=None)

# read data:
x_train = pd.read_csv('./data/ml2019spring-hw3/train_data.csv', header=None)
y_train = pd.read_csv('./data/ml2019spring-hw3/train_label.csv', header=None)
x_test = pd.read_csv('./data/ml2019spring-hw3/test_data.csv', header=None)

# normalize and one-hot:
x_train = x_train.values/255
x_test = x_test.values/255
y_train = np_utils.to_categorical(y_train.values.flatten(), 7)

# make sure the input data shape:
x_train = np.reshape(x_train, [-1, 48, 48, 1]).astype(np.float32)
x_test = np.reshape(x_test, [-1, 48, 48, 1]).astype(np.float32)

# validate:
pa4val = 0.2
x_valid = x_train[int(x_train.shape[0] * (1 - pa4val)): ]
y_valid = y_train[int(x_train.shape[0] * (1 - pa4val)): ]
y_train = y_train[0: int(x_train.shape[0] * (1 - pa4val))]
x_train = x_train[0: int(x_train.shape[0] * (1 - pa4val))]

# If subtract pixel mean is enabled
x_train_mean = np.mean(x_train, axis=0)
x_train -= x_train_mean
x_valid -= x_train_mean
x_test -= x_train_mean

# normalize and one-hot:
# train_data = train_data.values/255
# x_test = x_test.values/255
# train_label = np_utils.to_categorical(train_label.values.flatten(), 7)

# make sure the input data shape:
# train_data = np.reshape(train_data, [-1, 48, 48, 1])
x_test = np.reshape(x_test, [-1, 48, 48, 1])

# load model:
model = load_model("./logs/hw3/D20190927T2313/ResNet74v2_model_153.h5")
y_pred = model.predict(x_test)
y_pred_2 = np.argmax(y_pred, axis=1)
out_df = pd.DataFrame({'id': list(range(y_pred_2.shape[0])), 'label': y_pred_2})
out_df.to_csv("./logs/hw3/D20190927T2313/ResNet74v2_model_153.csv", index=False)
