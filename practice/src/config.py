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

    # Subtract the pixel mean of training data:
    SUBTRACT_PIXEL_MEAN = True

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

    # Save model weights each epoch or best only
    SAVE_BEST_ONLY = False

    # Resnetv2 layer depth:
    # depth should be 9n+2 (eg 56 or 110)
    RESNET_DEPTH = 56

    def __init__(self):
        self.list = None

    def display(self, print_out=True):
        self.list = ["Configurations:"]
        for a in dir(self):
            if not a.startswith("__") and not a.startswith("list") and not callable(getattr(self, a)):
                self.list.append("{:30} {}".format(a, getattr(self, a)))
        if print_out:
            for s in self.list:
                print(s)
        return self.list
