# -*- coding: utf-8 -*-
"""
Model configuration building with keras.
Created on 2019/9/29 下午 01:58
@author: Ivan Y.W.Chiu
"""


class Config(object):
    # The name of your work:
    NAME = 'None'

    # Number of class in dataset:
    N_CLASS = []

    # The shape of image in dataset:
    IMG_SHAPE = []

    # Learning rate:
    LR = 0.001

    # Epochs:
    EPOCHS = 20

    # Training batch size:
    BATCH_SIZE = 256

    # Validation rate:
    VALIDATION_RATE = 0.2

    # The initializer of kernels(weights) or bias in convolution or dense layer:
    # For kernels: 'glorot_normal', 'he_normal', truncated_normal(mean, std), ...
    KERNEL_INIT_METHOD = 'glorot_normal'
    # For bias: Constant value:
    BIAS_INIT_DEFAULT = 0

    # The arguments of convolution 2d layer:
    CONV2D_STRIDES = 1
    CONV2D_PADDING = "SAME"
    CONV2D_KERNEL_SIZE = (3, 3)

    # the arguments of max-pooling 2d layer:
    MAXPOOL2D_KERNEL_SIZE = (2, 2)
    MAXPOOL2D_PADDING = "SAME"
    MAXPOOL2D_STRIDES = 2

    # the arguments of upsampling 2d layer:
    UPSAMPLING_KERNEL_SIZE = 2
    UPSAMPLING_METHOD = 'nearest'

    # the dropout rate:
    DROPOUT_RATE = 0

    # The default activation function:
    ACTIVATION_FUNC = 'relu'

    def __init__(self):
        pass
