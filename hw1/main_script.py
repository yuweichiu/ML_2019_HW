# -*- coding: utf-8 -*-
"""
Homework 1 for ML course 2019
< Handcraft "linear regression" using Gradient Descent >

Created on : 2019/06/22,
@author: Ivan Chiu
"""

# import matplotlib
# matplotlib.use('Qt5Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# read csv file
os.chdir("./ml_2019_hw1")
train_df = pd.read_csv("./train_en.csv", encoding = "utf-8")
test_df = pd.read_csv("./test.csv", encoding = "utf-8", names=list(range(11)))

# data extraction
train_df_num = train_df.drop(columns=['date', 'Term'])
train_df_num.replace("NR", 0, inplace=True)
train_df_num.astype("float")

test_df.replace("NR", 0, inplace=True)
test_df[list(range(2, 11))] = test_df[list(range(2, 11))].astype(float)

# concat each day
train = np.zeros((18, int(24*4320/18)))
for i in range(4320//18):
    train[:, i*24: 24*(i+1)] = train_df_num.loc[i*18: 18*(i+1)-1].values

# gruop every ten hours (9 hours for train, 1 hours for validate)
i = 0
limit = train.shape[1]
sets = train.shape[1]-10+1
train_x = np.empty((sets, 18*9), dtype=float)
train_y = np.empty((sets, 1), dtype=float)
for s in range(sets):
    train_x[s, :] = train[:, s: s + 9].flatten()
    train_y[s, 0] = train[9, s + 9]

# Normalization:
mean_x = np.reshape(np.mean(train_x, axis=0), [1, -1])
std_x = np.reshape(np.std(train_x, axis=0), [1, -1])
mean_y = np.reshape(np.mean(train_y, axis=0), [1, -1])
std_y = np.reshape(np.std(train_y, axis=0), [1, -1])
train_xn = (train_x-mean_x)/std_x
train_yn = (train_y-mean_y)/std_y

# Initialize:
std = 0.078
mean = 0
lr = 6*1e-8
W = std * np.random.randn(18*9, 1) + mean
bias = np.ones(1)*0.01
loss = 0

epochl = []
llist = []

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.autoscale(enable=True)
ax.set_xlabel("epoch")
ax.set_ylabel("RMSE")
line = None
plt.ion()
plt.show()

# Training:
for i in range(200000):
    yhat = np.dot(train_xn, W)
    loss = train_yn - yhat
    grad = -2*np.dot(train_xn.T, loss)
    grad_b = -2*np.sum(loss)
    W = W - lr*grad
    bias = bias - lr*grad_b

    if i % 50 == 0:
        nl = np.power(np.sum(np.power(loss, 2))/loss.shape[0], 0.5)
        print("i="+str(i))
        print("loss="+str(nl))
        epochl.append(i)
        llist.append(nl)
        if line is None:
            line = ax.plot(epochl, llist)[0]
        else:
            line.set_xdata(epochl)
            line.set_ydata(llist)
            ax.set_xlim(epochl[0], epochl[-1])
            ax.set_ylim(min(llist), max(llist))
        fig.canvas.draw()
        plt.show()


# Testing
results = {}
id_list = []
test_x = np.empty((240, 18*9), dtype=float)
for id in range(240):
    id_list.append('id_'+str(id))
    curr_df = test_df[test_df[0] == 'id_'+str(id)]
    test_x[id, :] = curr_df.drop(columns=[0, 1]).values.flatten()

test_xn = (test_x - mean_x)/std_x
y_esn = np.dot(test_xn, W)
y_es = y_esn*std_y + mean_y
results['id'] = id_list
results['value'] = y_es.flatten()
results_df = pd.DataFrame(results)
results_df.to_csv('ans.csv', index=False)