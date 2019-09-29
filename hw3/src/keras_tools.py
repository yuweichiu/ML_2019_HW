# -*- coding: utf-8 -*-
"""
Created on 2019/7/21 上午 08:56
@author: Ivan Y.W.Chiu
"""

import tensorflow as tf
import numpy as np
import src.tf_tools as nn
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
        output = LeakyReLU(0.02)(x)
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


def model_summary(keras_model, param_dict=None, valid_acc_dict=None, print_out=True, save_dir=None):
    str_list = []
    s = '{0:25s} | {1:25s} | {2:20s} | {3:8s} | {4:8s} | {5:20s}'.format(
        'Name', 'Input_shape', 'Kernel size', 'Strides', 'Padding', 'Output_shape'
    )
    str_list.append(s)
    str_list.append("-"*121)
    for l in keras_model.layers:
        if l.name.split('_')[0] == 'conv2d':
            s = '{0:25s} | {1:25s} | {2:20s} | {3:8s} | {4:8s} | {5:20s}'.format(
                l.name, str(l.input_shape), str(l.kernel._keras_shape), str(l.strides), l.padding, str(l.output_shape)
            )
            str_list.append(s)
        elif l.name.split('_')[0] == 'conv2d' and l.name.split('_')[1] == 'transpose':
            s = '{0:25s} | {1:25s} | {2:20s} | {3:8s} | {4:8s} | {5:20s}'.format(
                l.name, str(l.input_shape), str(l.kernel._keras_shape), str(l.strides), l.padding, str(l.output_shape)
            )
            str_list.append(s)

        elif l.name.split('_')[0] == 'batch':
            s = '{0:25s} | {1:25s} | {2:20s} | {3:8s} | {4:8s} | {5:20s}'.format(
                l.name, str(l.input_shape), "", "", "", ""
            )
            str_list.append(s)
        elif l.name.split('_')[0] == 'activation':
            str0 = l.output.name.split("/")
            s = '{0:25s} | {1:25s} | {2:20s} | {3:8s} | {4:8s} | {5:20s}'.format(
                str0[0]+"/"+str0[-1], "", "", "", "", "")
            str_list.append(s)
        elif l.name.split('_')[0] == 'leaky':
            s = '{0:25s}'.format(
                l.name)
            str_list.append(s)
        elif (l.name.split('_')[0] == 'max') and (l.name.split('_')[1] == 'pooling2d'):
            s = '{0:25s} | {1:25s} | {2:20s} | {3:8s} | {4:8s} | {5:20s}'.format(
                l.name, str(l.input_shape), str(l.pool_size), str(l.strides), l.padding, str(l.output_shape)
            )
            str_list.append(s)
        elif (l.name.split('_')[0] == 'avg') and (l.name.split('_')[1] == 'pooling2d'):
            s = '{0:25s} | {1:25s} | {2:20s} | {3:8s} | {4:8s} | {5:20s}'.format(
                l.name, str(l.input_shape), str(l.pool_size), str(l.strides), l.padding, str(l.output_shape)
            )
            str_list.append(s)
        elif (l.name.split('_')[0] == 'up') and (l.name.split('_')[1] == 'sampling2d'):
            s = '{0:25s} | {1:25s} | {2:20s} | {3:8s} | {4:8s} | {5:20s}'.format(
                l.name, str(l.input_shape), "", str(l.size), "", str(l.output_shape)
            )
            str_list.append(s)
        elif l.name.split('_')[0] == 'flatten' or l.name.split('_')[0] == 'reshape':
            s = '{0:25s} | {1:25s} | {2:20s} | {3:8s} | {4:8s} | {5:20s}'.format(
                l.name, str(l.input_shape), "", "", "", str(l.output_shape))
            str_list.append(s)
        elif l.name.split('_')[0] == 'dense':
            s = '{0:25s} | {1:25s} | {2:20s} | {3:8s} | {4:8s} | {5:20s}'.format(
                l.name, str(l.input_shape), str(l.units), "", "", str(l.output_shape)
            )
            str_list.append(s)
        elif l.name.split('_')[0] == 'dropout':
            s = '{0:25s} | {1:25s} | {2:20s} | {3:8s} | {4:8s} | {5:20s}'.format(l.name, str(l.rate), "", "", "", "")
            str_list.append(s)
    str_list.append("-"*121)

    if param_dict:
        for key in sorted(param_dict.keys()):
            s = key + ": " + str(param_dict[key])
            str_list.append(s)
        str_list.append("-" * 121)

    if valid_acc_dict:
        str_list.append(str(len(valid_acc_dict)) + "-FOLD VALIDATION ACCURACY")
        acc = []
        for key in sorted(valid_acc_dict.keys()):
            acc.append(valid_acc_dict[key])
            s = key + ": " + "{0:7.4f}%".format(100*valid_acc_dict[key])
            str_list.append(s)
        mean_acc = np.mean(np.asarray(acc))
        str_list.append("AVG: {0:7.4f}%".format(100*mean_acc))

    if print_out:
        for s in str_list:
            print(s)
    if save_dir:
        with open(save_dir, 'w') as f:
            for s in str_list:
                f.write(s + "\n")

    return str_list