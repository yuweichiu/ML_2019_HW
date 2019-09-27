# -*- coding: utf-8 -*-
"""
Created on 2019/7/3 下午 08:38
@author: Ivan Y.W.Chiu
"""

# from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from keras.utils import np_utils
import numpy as np
import pandas as pd
import src.tf_tools as tft
import src.tf_models as tmd
import src.utils as utils
import time
import sys, os
import matplotlib.pyplot as plt

# read data:
x_train = pd.read_csv('./data/ml2019spring-hw3/train_data.csv', header=None)
y_train = pd.read_csv('./data/ml2019spring-hw3/train_label.csv', header=None)
x_test = pd.read_csv('./data/ml2019spring-hw3/test_data.csv', header=None)

# normalize and one-hot:
x_train = x_train.values/255
x_test = x_test.values/255
x_train = np.reshape(x_train, [-1, 48, 48, 1])
x_test = np.reshape(x_test, [-1, 48, 48, 1])
y_train = np_utils.to_categorical(y_train.values.flatten(), 7)
# seq = np.arange(0, x_train.shape[0])
#
# np.random.shuffle(seq)
# x_train = x_train[seq]
# y_train = y_train[seq]

# validate:
pa4val = 0.2
x_valid = x_train[int(x_train.shape[0] * (1 - pa4val)): ]
y_valid = y_train[int(x_train.shape[0] * (1 - pa4val)): ]
y_train = y_train[0: int(x_train.shape[0] * (1 - pa4val))]
x_train = x_train[0: int(x_train.shape[0] * (1 - pa4val))]
seq = np.arange(0, x_train.shape[0])

# build model:
lr = 0.001
epoch = 100
batch = 256

train_bl = utils.batch_index(batch, x_train.shape[0])
valid_bl = utils.batch_index(batch, x_valid.shape[0])
loss_train = []
loss_valid = []
acc_train = []
acc_valid = []
max_val = 0

# Session:
tf.reset_default_graph()  # reset the graph

# get model:
model = tmd.init_model([48, 48, 1], 7)
end_points = tmd.VGG(model)
model = tmd.compile_model(model, end_points, optimizer='Adam', lr=0.001)
NET = model

# start session and saver:
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

# create project
tn = time.localtime()
project = "./logs/hw3/D{0:4d}{1:02d}{2:02d}T{3:02d}{4:02d}".format(tn[0], tn[1], tn[2], tn[3], tn[4])
os.mkdir(project)

t_0 = time.time()
for ep in range(epoch):
    lst = 0
    lsv = 0
    acct = 0
    accv = 0
    np.random.shuffle(seq)
    x_train = x_train[seq]
    y_train = y_train[seq]
    t_1 = time.time()
    print('\nEpoch: {0:d}/{1:d}'.format(ep+1, epoch))
    countt = 1
    for btid, bt in enumerate(train_bl):
        if bt == train_bl[-1]:
            batch_x, batch_y = x_train[bt:], y_train[bt:]
            cur_d = x_train.shape[0]
        else:
            batch_x, batch_y = x_train[bt: train_bl[btid+1]], y_train[bt: train_bl[btid+1]]
            cur_d = train_bl[btid+1]
        train_dict = {NET['xs']: batch_x, NET['ys']: batch_y, NET['drop_rate']: 0.5, NET['is_training']: True}
        _, lsb, accb = sess.run([NET['train_op'], NET['avg_loss'], NET['accuracy']], feed_dict=train_dict)
        lst = lst + lsb
        acct = acct + accb
        lstp, acctp = lst / countt, acct / countt
        t_2 = time.time()
        sys.stdout.write(
            '\r{0:10s} Data: {1:5d}/{2:5d} | loss : {3:>9.7f} | Accuracy : {4:>5.4f} | Time: {5:>7.4f} sec'.format(
                "[Train]", cur_d, x_train.shape[0], lstp, acctp, t_2 - t_1
            ))
        countt += 1
    loss_train.append(lstp)
    acc_train.append(acctp)

    sys.stdout.write("\n")
    countv = 1
    for btid, bt in enumerate(valid_bl):
        if bt == valid_bl[-1]:
            batch_x, batch_y = x_valid[bt:], y_valid[bt:]
            cur_d = x_valid.shape[0]
        else:
            batch_x, batch_y = x_valid[bt: valid_bl[btid+1]], y_valid[bt: valid_bl[btid+1]]
            cur_d = valid_bl[btid+1]

        valid_dict = {NET['xs']: batch_x, NET['ys']: batch_y, NET['is_training']: False}
        lsb, accb = sess.run([NET['avg_loss'], NET['accuracy']], feed_dict=valid_dict)
        lsv = lsv + lsb
        accv = accv + accb
        lsvp, accvp = lsv/countv, accv/countv
        t_3 = time.time()
        sys.stdout.write(
            '\r{0:10s} Data: {1:5d}/{2:5d} | loss : {3:>9.7f} | Accuracy : {4:>5.4f} | Time: {5:>7.4f} sec'.format(
                "[Validate]", cur_d, x_valid.shape[0], lsvp, accvp, t_3 - t_2
            ))
        countv += 1
    loss_valid.append(lsvp)
    acc_valid.append(accvp)
    if acc_valid[-1] > max_val:
        sys.stdout.write(
            '\nSaving model...Validation accuracy increased to {0:8.5f}\n'.format(acc_valid[-1])
            )
        saver.save(sess, project + '/best_model')
        max_val = acc_valid[-1]

t_4 = time.time()
print("\nTotal Time Cost: {0:5f}".format(t_4-t_0))

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(range(epoch), acc_train, label='Train')
ax.plot(range(epoch), acc_valid, label='Validate')
ax.legend()
plt.show()




