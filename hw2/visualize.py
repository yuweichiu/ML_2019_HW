# -*- coding: utf-8 -*-
"""
Created on : 2019/06/27,
@author: Ivan Chiu
"""

import os
import matplotlib.pyplot as plt
import numpy as np

os.chdir("./ml_2019_hw2")
l_tr = np.loadtxt("loss_train.txt")
l_va = np.loadtxt("loss_valid.txt")
l_tr_ada = np.loadtxt("loss_train_adagrad.txt")
l_va_ada = np.loadtxt("loss_valid_adagrad.txt")
epoch = list(range(len(l_tr)))

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(epoch, l_tr, label="Train")
ax.plot(epoch, l_va, label="Validate")
ax.plot(epoch, l_tr_ada, label="Train + Adagrad")
ax.plot(epoch, l_va_ada, label="Validate + Adagrad")
ax.legend()
plt.show()
