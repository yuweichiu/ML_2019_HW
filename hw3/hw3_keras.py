# -*- coding: utf-8 -*-
"""
The main script for hw3 in keras.
Created on 2019/9/29 下午 05:01
@author: Ivan Y.W.Chiu
"""

from keras.datasets import cifar10
import hw3.src.model as model

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
#  Configurations
############################################################


############################################################
#  Datasets
############################################################
n_class = 10
img_shape = [32, 32, 3]
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# Make sure the shape is N, H, W, C:
x_train = np.reshape(x_train, [-1] + img_shape)
x_test = np.reshape(x_test, [-1] + img_shape)

# Normalization:
x_train, x_test = x_train.astype('float32')/255, x_test.astype('float32')/255

# One-hot encoding:
y_train = np_utils.to_categorical(y_train, n_class).astype('float32')
y_test = np_utils.to_categorical(y_test, n_class).astype('float32')


############################################################
#  Train
############################################################


############################################################
#  Classification
############################################################



if __name__ == '__main__':
    import argparse

