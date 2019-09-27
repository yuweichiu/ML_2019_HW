# -*- coding: utf-8 -*-
"""
Created on 2019/7/3 下午 02:23
@author: Ivan Y.W.Chiu
"""

import numpy as np

data = []
with open('/media/yuwei/Data/YuWei/NTU/ML_HungYiLee/ml2019spring-hw3/train.csv', 'r') as f:
    data = f.readlines()
data.pop(0)  # drop first line "label, feature"
y_train = []
x_train = []
for did, d in enumerate(data):
    temp = d.split(",")
    lb = temp[0]
    dt = temp[1].split("\n")[0]
    y_train.append(lb)
    x_train.append(dt.split(" "))

x_train = np.asarray(x_train, dtype=int)
y_train = np.asarray(y_train, dtype=int)
np.savetxt('/media/yuwei/Data/YuWei/NTU/ML_HungYiLee/ml2019spring-hw3/train_data.csv', x_train, fmt='%d', delimiter=",")
np.savetxt('/media/yuwei/Data/YuWei/NTU/ML_HungYiLee/ml2019spring-hw3/train_label.csv', y_train, fmt='%d')

data = []
with open('/media/yuwei/Data/YuWei/NTU/ML_HungYiLee/ml2019spring-hw3/test.csv', 'r') as f:
    data = f.readlines()
data.pop(0)  # drop first line "label, feature"
y_test = []
x_test = []
for did, d in enumerate(data):
    temp = d.split(",")
    # lb = temp[0]
    dt = temp[1].split("\n")[0]
    # y_test.append(lb)
    x_test.append(dt.split(" "))

x_test = np.asarray(x_test, dtype=int)
np.savetxt('/media/yuwei/Data/YuWei/NTU/ML_HungYiLee/ml2019spring-hw3/test_data.csv', x_test, fmt='%d', delimiter=",")
