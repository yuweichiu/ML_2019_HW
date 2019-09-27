# -*- coding: utf-8 -*-
"""
Homework 2 on Machine Learning Course
Created on : 2019/6/26
@author: Ivan Chiu
"""

import os
import pandas as pd
import numpy as np

#read data
df = pd.read_csv("./hw2/train.csv", encoding='utf-8')
df2 = pd.read_csv("./hw2/test.csv", encoding='utf-8')

# cleaning
df.replace(" ?", np.nan, inplace=True)
df2.replace(" ?", np.nan, inplace=True)
df = df.dropna(axis=0)
df2 = df2.dropna(axis=0)

# verify dtype
df['income'].astype(object)
df.loc[df['income']==' <=50K', 'income'] = "0"
df.loc[df['income']==' >50K', 'income'] = "1"
df['income'].astype(float)

# one-hot encoding
col2change = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']
change_dict = {}
for c in col2change:
    one_hot = pd.get_dummies(df[c])
    change_dict[c] = list(one_hot.columns)
    one_hot2 = pd.get_dummies(df2[c], columns=list(one_hot.columns))
    df.drop(columns=c, inplace=True)
    df2.drop(columns=c, inplace=True)
    df = pd.concat([df, one_hot], axis=1)
    df2 = pd.concat([df2, one_hot2], axis=1)

df = df.astype(float)
df2 = df2.astype(float)

# Define x, y data
# x_train = df[list(df.columns)[:-1]].values
x_train_r = df.drop(columns='income').values
y_train_r = df['income'].values
x_test_r = df2.values

# Normalization
max_x = np.reshape(x_train_r.max(axis=0), (1, x_train_r.shape[1]))
min_x = np.reshape(x_train_r.min(axis=0), (1, x_train_r.shape[1]))
x_train_n = np.divide(np.subtract(x_train_r, min_x), np.subtract(max_x, min_x))

# Train and validate:
def train_valid(X, Y, rate):
    number = X.shape[0]
    num_series = np.asarray(list(range(0, number)))
    # np.random.shuffle(num_series)
    train_id = num_series[:int(number*(1-rate))]
    valid_id = num_series[int(number*(1-rate)):]
    x_train, y_train = X[train_id, :], Y[train_id]
    x_valid, y_valid = X[valid_id, :], Y[valid_id]
    return x_train, y_train, x_valid, y_valid


def sigmoid(z):
    return np.clip(1 / (1 + np.exp(-z)), 1e-6, 1-1e-6)


def cross_entropy(y_pred, y_target):
    return -np.dot(y_target, np.log(np.clip(y_pred, 1e-10, 1)))-np.dot(1-y_target, np.log(np.clip(1-y_pred, 1e-10, 1)))


def accuracy(y_pred, y_target):
    y_pred = np.round(y_pred)
    return 100*np.sum(y_target == y_pred) / len(y_pred)

x_train, y_train, x_valid, y_valid = train_valid(x_train_n, y_train_r, 0.1155)

# Initialize weight and bias:
std = 0.098
mean = 0
lr = 0.4
lamb = 0.001  # regularization
w = std * np.random.randn(x_train_n.shape[1]) + mean
b = np.ones(1)*0  # one node only have one bias value
loss = 0
batch_size = 32
step = 1
loss_train = []
loss_valid = []
gwh = 0
gbh = 0

for e in range(500):
    for bid in range(int(np.floor(len(y_train)/batch_size))):
        X = x_train[bid*batch_size: (bid+1)*batch_size]
        Y = y_train[bid*batch_size: (bid+1)*batch_size]
        Z = np.add(np.matmul(X, w), b)
        Y_hat = sigmoid(Z)
        error = Y - Y_hat
        grad_w = -np.mean(np.multiply(error.T, X.T), axis=1) + lamb*w
        grad_b = -np.mean(error)
        # Adagrad:
        if step == 1:
            w = w - lr/np.sqrt(step) * grad_w
            b = b - lr/np.sqrt(step) * grad_b
        else: 
            w = w - lr/np.sqrt(step) * grad_w /(gwh/step)**0.5
            b = b - lr/np.sqrt(step) * grad_b /(gbh/step)**0.5
        # w = w - lr/np.sqrt(step) * grad_w
        # b = b - lr/np.sqrt(step) * grad_b
        step += 1
        gwh = gwh + np.sum(np.square(grad_w))
        gbh = gbh + np.square(grad_b)

    Y_tr_hat = sigmoid(np.add(np.matmul(x_train, w), b))
    loss_tr = (cross_entropy(Y_tr_hat, y_train) + lamb * np.sum(np.square(w)))/len(y_train)
    acc_tr = accuracy(Y_tr_hat, y_train)
    loss_train.append(loss_tr)

    Z_val = np.add(np.matmul(x_valid, w), b)
    Y_val_hat = sigmoid(Z_val)
    loss_val = (cross_entropy(Y_val_hat, y_valid) + lamb * np.sum(np.square(w)))/len(y_valid)
    acc_val = accuracy(Y_val_hat, y_valid)
    loss_valid.append(loss_val)
    print("Train Loss = {0:6f}; Train Acc = {1:6f} %; Validate Loss = {2:6f}; Validate Acc = {3:6f} %".format(loss_tr, acc_tr, loss_val, acc_val))


x_test_n = np.divide(np.subtract(x_test_r, min_x), np.subtract(max_x, min_x))

# np.savetxt('./hw2/loss_train.txt', loss_train)
# np.savetxt('./hw2/loss_valid.txt', loss_valid)









