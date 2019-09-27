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
import src.mynet as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from keras.models import load_model

# read data:
# train_data = pd.read_csv(r'D:\YuWei\NTU\ML_HungYiLee\ml2019spring-hw3\train_data.csv', header=None)
# train_label = pd.read_csv(r'D:\YuWei\NTU\ML_HungYiLee\ml2019spring-hw3\train_label.csv', header=None)
test_data = pd.read_csv(r'D:\YuWei\NTU\ML_HungYiLee\ml2019spring-hw3\test_data.csv', header=None)

# normalize and one-hot:
# train_data = train_data.values/255
test_data = test_data.values/255
# train_label = np_utils.to_categorical(train_label.values.flatten(), 7)

# make sure the input data shape:
# train_data = np.reshape(train_data, [-1, 48, 48, 1])
test_data = np.reshape(test_data, [-1, 48, 48, 1])

# load model:
model = load_model("validate_190708_02/network_confirm/best_model_099603.h5")
y_pred = model.predict(test_data)
y_pred_2 = np.argmax(y_pred, axis=1)
out_df = pd.DataFrame({'id': list(range(y_pred_2.shape[0])), 'label': y_pred_2})
out_df.to_csv("ans_06.csv", index=False)
