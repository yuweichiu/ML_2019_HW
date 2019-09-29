# -*- coding: utf-8 -*-
"""
Collect built models on tensorflow.
Created on 2019/07/12
@author: Ivan Y.W.Chiu
"""

import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import src.tf_tools as nn
from tensorflow.contrib import layers as ly
from tensorflow.contrib.framework.python.ops import arg_scope


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


def init_model_new():
    """
    Initialize tf model.
    :param data_shape: [width, height. channel]
    :param category: number of classess
    :return: model
    """
    # xs = tf.placeholder(tf.float32, [None, data_shape[0], data_shape[1], data_shape[2]], name='inputs')
    # ys = tf.placeholder(tf.float32, [None, category], name='outputs')
    drop_rate = tf.placeholder(tf.float32, name='rate')
    is_training = tf.placeholder(tf.bool, name='is_training')
    # model_input = xs
    model = {
        'drop_rate': drop_rate,
        'is_training': is_training,
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


def VGG_new(model, input, n_class):
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
            net = nn.conv2d('conv1_1', input, 64)
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
            outputs = nn.dense('output', net, n_class, activation='softmax', norm=False)
            end_points['Prediction'] = outputs
        return end_points


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


def compile_model_new(model, end_points, label, use_aux=False, optimizer='Adam', lr=0.0001):
    is_training = model['is_training']
    each_loss = nn.each_loss(end_points, label, is_training, aux=use_aux)
    avg_loss = nn.avg_loss(each_loss)
    ACC = nn.accuracy(end_points['Prediction'], label)
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
