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
        self.n_class = []
        self.img_shape = []
        self.val_rate = []
        self.use_val = True
        self.validation_set = None

    def prepare(self):
        # Make sure the shape is N, H, W, C:
        self.x_data = np.reshape(self.x_data, [-1] + self.img_shape).astype('float32')

        # One-hot encoding:
        self.y_data = np_utils.to_categorical(self.y_data, self.n_class).astype('float32')


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