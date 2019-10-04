# -*- coding: utf-8 -*-
"""
Utilities for training network.

Created on : 2019/9/27
@author: Ivan Chiu
"""

import numpy as np
from keras.utils import np_utils
from keras.datasets import cifar10


class Dataset(object):
    def __init__(self):
        self.n_class = []  # number of class from config
        self.img_shape = []  # the shape of image from config
        self.use_val = None  # use validation set or not, if validation rate=0, set as false.
        self.validation_set = None  # prepare for generating validation set in training process.

    def prepare(self):
        # Make sure the shape is N, H, W, C:
        self.x_data = np.reshape(self.x_data, [-1] + self.img_shape).astype('float32')

        # One-hot encoding:
        self.y_data = np_utils.to_categorical(self.y_data, self.n_class).astype('float32')

    def split(self):
        """
        Split the dataset as training set or validation set
        :param usage (str): 'training_set' or 'validation_set'
        :return:
        """
        val_rate = self.config.VALIDATION_RATE
        total = self.x_data.shape[0]

        # For training set:
        self.x_train = self.x_data[0: int(total * (1 - val_rate))]
        self.y_train = self.y_data[0: int(total * (1 - val_rate))]
        print('Training set: {:d}'.format(int(total * (1 - val_rate))))

        # For validation set:
        # If validation rate is 0, means don't use validation set.
        if val_rate == 0:
            self.use_val = False
            print("Validation set: N/A")
            pass
        else:
            self.use_val = True
            self.x_val = self.x_data[int(total * (1 - val_rate)): ]
            self.y_val = self.y_data[int(total * (1 - val_rate)): ]
            print('Validation set: {:d}'.format(int(total * val_rate)))
            

def batch_index(b_size, total):
    lid = []
    for i in range(total//b_size):
        lid.append(i*b_size)
        # print(i, i*b_size)
    if (i + 1)*b_size < total:
        i = i + 1
        lid.append(i*b_size)
    else:
        pass
    return lid


def N_Fold_Validate(n_splits, num_data):
    splits = n_splits + 1
    id_max = num_data - 1
    seq = np.linspace(0, id_max, splits, endpoint=True, dtype=int)
    # seq[-1] = seq[-1]+1
    nf_list = []
    for i in range(n_splits):
        start = seq[i]
        end = seq[i+1]
        nf_list.append(np.arange(start, end).tolist())

    nf_list_f = []
    for id, nf in enumerate(nf_list):
        temp = nf_list.copy()
        temp.pop(id)
        train = []
        for t in temp:
            train = train + t
        nf_list_f.append([train, nf])

    return nf_list_f