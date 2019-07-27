# -*- coding: utf-8 -*-
"""
Created on 2019/07/12
@author: Ivan Y.W.Chiu
"""

import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import src.nntools as nn
import src.keras_tools as knt
from tensorflow.contrib import layers as ly
from tensorflow.contrib.framework.python.ops import arg_scope
from keras.models import Sequential, Model
from keras.optimizers import SGD, Adam
from keras_preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPool2D
from keras.layers import Flatten, LeakyReLU, UpSampling2D, Reshape
from keras.layers.normalization import BatchNormalization
from keras.initializers import Constant, truncated_normal
from keras.optimizers import SGD, Adam
from keras import backend as K



def init_model(data_shape, category):
    """
    Initialize tf model.
    :param data_shape: [width, height. channel]
    :param category: number of classess
    :return: model
    """
    xs = tf.placeholder(tf.float32, [None, data_shape[0], data_shape[1], data_shape[2]], name='inputs')
    ys = tf.placeholder(tf.float32, [None, category], name='outputs')
    drop_rate = tf.placeholder(tf.float32, name='rate')
    is_training = tf.placeholder(tf.bool, name='is_training')
    model_input = xs
    model = {
        'data_shape': data_shape,
        'category': category,
        'xs': xs,
        'ys': ys,
        'drop_rate': drop_rate,
        'is_training': is_training,
        'input': model_input
    }
    return model

def VGG(model):
    """ VGG - tiny """
    end_points = {}
    is_training = model['is_training']
    dp = model['drop_rate']
    with tf.variable_scope("Network"):
        s = '{0:20s} | {1:25s} | {2:20s} | {3:8s} | {4:8s} | {5:20s} | {6:20s}'.format(
            'Name', 'Input_shape', 'Kernel size', 'Strides', 'Padding', 'Output_shape', 'Parameter'
        )
        print(s)
        print('-'*144)
        with arg_scope([nn.conv2d], kernel_size=[3, 3], stride=1,
                       is_training=is_training, activation='relu'):
            net = nn.conv2d('conv1_1', model['input'], 64)
            net = nn.conv2d('conv1_2', net, 64)
            net = nn.maxpool2d('max_pool_1', net, [2, 2], stride=2)
            net = nn.conv2d('conv2_1', net, 128)
            net = nn.conv2d('conv2_2', net, 128)
            net = nn.maxpool2d('max_pool_2', net, [2, 2], stride=2)
            net = nn.conv2d('conv3_1', net, 256)
            net = nn.conv2d('conv3_2', net, 256)
            net = nn.maxpool2d('max_pool_3', net, [2, 2], stride=2)
            net = nn.conv2d('conv4_1', net, 512)
            net = nn.conv2d('conv4_2', net, 512)
            net = nn.maxpool2d('max_pool_4', net, [2, 2], stride=2)
            net = nn.flatten('Flatten', net)
        with arg_scope([nn.dense, nn.dropout], is_training=is_training):
            net = nn.dense('dense_1', net, 1024, activation='relu')
            net = nn.dropout('dropout_1', net, dp)
            net = nn.dense('dense_2', net, 512, activation='relu')
            net = nn.dropout('dropout_2', net, dp)
            outputs = nn.dense('output', net, model['category'], activation='softmax', norm=False)
            end_points['Prediction'] = outputs
        return end_points


def k_autoencoder3(latent_dim, folder):
    inputs = knt.md_input((32, 32, 3))
    net = knt.conv2d(inputs, 64, (3, 3))  # 32,32,64
    net = knt.activation(net, "relu")
    net = knt.conv2d(net, 64, (3, 3))  # 32,32,64
    net = knt.activation(net, "relu")
    net = knt.maxpool2d(net)  # 16,16,64
    net = knt.conv2d(net, 128, (3, 3))  # 16,16,128
    net = knt.activation(net, "relu")
    net = knt.conv2d(net, 128, (3, 3))  # 16,16,128
    net = knt.activation(net, "relu")
    net = knt.maxpool2d(net)  # 8,8,128
    net = knt.conv2d(net, 256, (3, 3))  # 8,8,256
    net = knt.activation(net, "relu")
    net = knt.conv2d(net, 256, (3, 3))
    net = knt.activation(net, "relu")
    net = knt.maxpool2d(net)  # 4,4,256
    net = knt.conv2d(net, 512, (3, 3))  # 4,4,512
    net = knt.activation(net, "relu")
    net = knt.conv2d(net, 512, (3, 3))
    net = knt.flatten(net)  # 4*4*512
    net = knt.dense(net, 2048)  # 1024
    net = knt.activation(net, "relu")
    net = knt.dense(net, latent_dim)
    encode = knt.activation(net, "relu")  # 23
    net = knt.dense(encode, 2048)
    net = knt.activation(net, "relu")
    net = knt.dense(net, 4 * 4 * 512)
    net = knt.activation(net, "relu")
    net = knt.reshape(net, (4, 4, 512))  # 4,4,512
    net = knt.deconv2d(net, 512, (3, 3))
    net = knt.activation(net, "relu")
    net = knt.deconv2d(net, 512, (3, 3))
    net = knt.upsampling(net, 2)  # 8,8,512
    net = knt.deconv2d(net, 256, (3, 3))  # 8,8,256
    net = knt.activation(net, "relu")
    net = knt.deconv2d(net, 256, (3, 3))
    net = knt.upsampling(net, 2)  # 16,16,256
    net = knt.deconv2d(net, 128, (3, 3))  # 16,16,128
    net = knt.activation(net, "relu")
    net = knt.deconv2d(net, 128, (3, 3))
    net = knt.activation(net, "relu")
    net = knt.upsampling(net, 2)  # 32,32,128
    net = knt.deconv2d(net, 64, (3, 3))  # 32,32,64
    net = knt.activation(net, "relu")
    net = knt.deconv2d(net, 64, (3, 3))
    net = knt.activation(net, "relu")
    net = knt.deconv2d(net, 3, (3, 3))  # 32,32,3
    decode = knt.activation(net, "sigmoid")
    auto_encoder = Model(inputs, decode)
    net_struct = nn.model_summary(auto_encoder, print_out=True, save_dir=folder + "/model_summary.txt")

    encoder = Model(inputs, encode)
    encoded_input = knt.md_input(shape=(latent_dim,))
    decoding = auto_encoder.layers[24](encoded_input)
    for layer in auto_encoder.layers[25:]:
        decoding = layer(decoding)
    decoder = Model(encoded_input, decoding)

    return auto_encoder, encoder, decoder



def compile_model(model, end_points, use_aux=False, optimizer='Adam', lr=0.0001):
    ys = model['ys']
    is_training = model['is_training']
    each_loss = nn.each_loss(end_points, ys, is_training, aux=use_aux)
    avg_loss = nn.avg_loss(each_loss)
    ACC = nn.accuracy(end_points['Prediction'], ys)
    # Define training step
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        if optimizer=='Adam':
            train_step = tf.train.AdamOptimizer(lr).minimize(avg_loss)
        elif optimizer=='SGD':
            train_step = tf.train.GradientDescentOptimizer(lr).minimize(avg_loss)
    model['train_op'] = train_step
    model['predict'] = end_points['Prediction']
    model['each_loss'] = each_loss
    model['avg_loss'] = avg_loss
    model['accuracy'] = ACC
    return model
