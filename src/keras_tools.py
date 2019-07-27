# -*- coding: utf-8 -*-
"""
Created on 2019/7/21 上午 08:56
@author: Ivan Y.W.Chiu
"""

import tensorflow as tf
import numpy as np
import src.nntools as nn
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPool2D, AvgPool2D
from keras.layers import Flatten, LeakyReLU, UpSampling2D, Reshape, Input
from keras.layers.normalization import BatchNormalization
from keras.initializers import Constant, truncated_normal
from keras import backend as K

kernel_init_default = 'glorot_normal'

def md_input(shape):
    return Input(shape=shape)


def conv2d(x, depth, kernel_size, stride=1, padding="SAME", kernel_init=kernel_init_default, bias_init=0):
    output = Conv2D(depth, kernel_size=kernel_size, strides=stride, padding=padding,
                    kernel_initializer=kernel_init,
                    bias_initializer=Constant(bias_init))(x)
    return output


def deconv2d(x, depth, kernel_size, stride=1, padding="SAME", kernel_init=kernel_init_default, bias_init=0):
    output = Conv2DTranspose(depth, kernel_size=kernel_size, strides=stride, padding=padding,
                    kernel_initializer=kernel_init,
                    bias_initializer=Constant(bias_init))(x)
    return output


def batch_norm(x):
    output = BatchNormalization()(x)
    return output

def activation(x, fn=None):
    if fn == 'relu':
        output = Activation("relu")(x)
    elif fn == 'softmax':
        output = Activation("softmax")(x)
    elif fn == 'sigmoid':
        output = Activation("sigmoid")(x)
    elif fn == 'LeakyReLU':
        output = LeakyReLU(0.02)
    else: output = x
    return output


def maxpool2d(x, kernel_size=(2, 2), stride=2, padding="SAME"):
    output = MaxPool2D(pool_size=kernel_size, strides=stride, padding=padding)(x)
    return output


def avgpool2d(x, kernel_size, stride=1, padding="SAME"):
    output = AvgPool2D(pool_size=kernel_size, strides=stride, padding=padding)(x)
    return output


def upsampling(x, up_size, method='nearest'):
    output = UpSampling2D(size=(up_size, up_size), interpolation=method)(x)
    return output


def dense(x, units, kernel_init=kernel_init_default, bias_init=0):
    output = Dense(units, kernel_initializer=kernel_init, bias_initializer=Constant(bias_init))(x)
    return output


def dropout(x, rate):
    output = Dropout(rate)(x)
    return output


def flatten(x):
    output = Flatten()(x)
    return output


def reshape(x, shape):
    output = Reshape(shape)(x)
    return output







