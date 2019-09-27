# -*- coding: utf-8 -*-
"""
Try the tf.data as the data input pipeline
(Done)

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

# Session:
tf.reset_default_graph()  # reset the graph

xs = tf.placeholder(tf.float32, [None, 48, 48, 1], name='inputs')
ys = tf.placeholder(tf.float32, [None, 7], name='outputs')

train_val_dataset = tf.data.Dataset.from_tensor_slices((xs, ys))
train_val_dataset = train_val_dataset.batch(batch)
iterator = tf.data.Iterator.from_structure(train_val_dataset.output_types, train_val_dataset.output_shapes)
data_x, data_y = iterator.get_next()
train_val_dataset = train_val_dataset.prefetch(1)

loss_train = []
loss_valid = []
acc_train = []
acc_valid = []
max_val = 0

# get model:
model = tmd.init_model_new()
end_points = tmd.VGG_new(model, data_x, 7)
model = tmd.compile_model_new(model, end_points, data_y, optimizer='Adam', lr=0.001)
NET = model

train_iterator = iterator.make_initializer(train_val_dataset)
valid_iterator = iterator.make_initializer(train_val_dataset)

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
    print('\nEpoch: {0:d}/{1:d}'.format(ep+1, epoch))
    countt = 1
    sess.run(train_iterator, feed_dict={xs: x_train, ys: y_train})
    t_1 = time.time()
    try:
        while True:
            _, lsb, accb = sess.run([NET['train_op'], NET['avg_loss'], NET['accuracy']],
                                    feed_dict={NET['is_training']: True, NET['drop_rate']: 0.5})
            lst = lst + lsb
            acct = acct + accb
            lstp, acctp = lst / countt, acct / countt
            t_2 = time.time()
            sys.stdout.write(
                '\r{0:10s} Data: {1:5d}/{2:5d} | loss : {3:>9.7f} | Accuracy : {4:>5.4f} | Time: {5:>7.4f} sec'.format(
                    "[Train]", 0, x_train.shape[0], lstp, acctp, t_2 - t_1
                ))
            countt += 1
    except tf.errors.OutOfRangeError:
        pass

    loss_train.append(lstp)
    acc_train.append(acctp)

    sys.stdout.write("\n")
    countv = 1
    sess.run(train_iterator, feed_dict={xs: x_valid, ys: y_valid})
    t_2 = time.time()
    try:
        while True:
            lsb, accb = sess.run([NET['avg_loss'], NET['accuracy']],
                                 feed_dict={NET['is_training']: False})
            lsv = lsv + lsb
            accv = accv + accb
            lsvp, accvp = lsv / countv, accv / countv
            t_3 = time.time()
            sys.stdout.write(
                '\r{0:10s} Data: {1:5d}/{2:5d} | loss : {3:>9.7f} | Accuracy : {4:>5.4f} | Time: {5:>7.4f} sec'.format(
                    "[Validate]", 0, x_valid.shape[0], lsvp, accvp, t_3 - t_2
                ))
            countv += 1
    except tf.errors.OutOfRangeError:
        pass

    loss_valid.append(lsvp)
    acc_valid.append(accvp)
    if ep > 50:
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




