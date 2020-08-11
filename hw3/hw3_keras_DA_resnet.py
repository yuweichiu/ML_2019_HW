# -*- coding: utf-8 -*-
"""
Resnet v2 in keras
Created on : 2019/8/14
@author: Ivan Chiu
@ref: https://keras.io/examples/cifar10_resnet/
"""

from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.utils import np_utils
from keras.models import Model
from keras.datasets import cifar10
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

############################################################
#  Session Setting
############################################################
# If you face the error about convolution layer,
# use this block to enable the memory usage of GPU growth.
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras


# Training parameters
batch_size = 32  # orig paper trained all networks with batch_size=128
epochs = 200
data_augmentation = True
num_classes = 10

# Subtracting pixel mean improves accuracy
subtract_pixel_mean = True

# Model parameter
n = 6  # i.e. num_res_blocks

# Computed depth from supplied model parameter n
depth = n * 12 + 2

# Model name, depth and version
model_type = 'ResNet%dv2' % depth

# read data:
x_train = pd.read_csv('./data/ml2019spring-hw3/train_data.csv', header=None)
y_train = pd.read_csv('./data/ml2019spring-hw3/train_label.csv', header=None)
x_test = pd.read_csv('./data/ml2019spring-hw3/test_data.csv', header=None)

# normalize and one-hot:
x_train = x_train.values/255
x_test = x_test.values/255
y_train = np_utils.to_categorical(y_train.values.flatten(), 7)

# make sure the input data shape:
x_train = np.reshape(x_train, [-1, 48, 48, 1]).astype(np.float32)
x_test = np.reshape(x_test, [-1, 48, 48, 1]).astype(np.float32)

# validation:
pa4val = 0.2
if pa4val != 0:
    x_valid = x_train[int(x_train.shape[0] * (1 - pa4val)): ]
    y_valid = y_train[int(x_train.shape[0] * (1 - pa4val)): ]
    y_train = y_train[0: int(x_train.shape[0] * (1 - pa4val))]
    x_train = x_train[0: int(x_train.shape[0] * (1 - pa4val))]

# Input image dimensions.
input_shape = x_train.shape[1:]

# If subtract pixel mean is enabled
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    if pa4val != 0:
        x_valid -= x_train_mean

print(x_train.shape[0], 'train samples')
if pa4val != 0:
    print(x_valid.shape[0], 'validation samples')

# Convert class vectors to binary class matrices.
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_valid = keras.utils.to_categorical(y_valid, num_classes)


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v2(input_shape, depth, num_classes=7):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 48x48,  16
    stage 0: 48x48,  64
    stage 1: 24x24, 128
    stage 2: 12x12, 256
    stage 3: 6 x 6, 512

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (e.g. CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 12 != 0:
        raise ValueError('depth should be 12n+2')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 12)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(4):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=6)(x)  # the dimension of last layer
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


model = resnet_v2(input_shape=input_shape, depth=depth)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])
model.summary()
print(model_type)

# create project
tn = time.localtime()
project = "./logs/hw3/D{0:4d}{1:02d}{2:02d}T{3:02d}{4:02d}".format(tn[0], tn[1], tn[2], tn[3], tn[4])
os.mkdir(project)

# Prepare model model saving directory.
# save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = '%s_model_{epoch:03d}.h5' % model_type
filepath = os.path.join(project, model_name)

# Prepare callbacks for model saving and for learning rate adjustment.
if pa4val != 0.2:
    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True)
else:
    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='acc',
                                 verbose=1,
                                 save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

tensorboard = TensorBoard(log_dir=project, histogram_freq=0, write_graph=True, write_images=False)

callbacks = [checkpoint, lr_reducer, lr_scheduler, tensorboard]

# Run training, with or without data augmentation.
if not data_augmentation:
    print('Not using data augmentation.')
    if pa4val != 0:
        hist = model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_valid, y_valid),
            shuffle=True,
            callbacks=callbacks
        )
    else:
        hist = model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            shuffle=True,
            callbacks=callbacks
        )

else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    # datagen = ImageDataGenerator(
    #     # set input mean to 0 over the dataset
    #     featurewise_center=False,
    #     # set each sample mean to 0
    #     samplewise_center=False,
    #     # divide inputs by std of dataset
    #     featurewise_std_normalization=False,
    #     # divide each input by its std
    #     samplewise_std_normalization=False,
    #     # apply ZCA whitening
    #     zca_whitening=False,
    #     # epsilon for ZCA whitening
    #     zca_epsilon=1e-06,
    #     # randomly rotate images in the range (deg 0 to 180)
    #     rotation_range=0,
    #     # randomly shift images horizontally
    #     width_shift_range=0.1,
    #     # randomly shift images vertically
    #     height_shift_range=0.1,
    #     # set range for random shear
    #     shear_range=0.,
    #     # set range for random zoom
    #     zoom_range=0.,
    #     # set range for random channel shifts
    #     channel_shift_range=0.,
    #     # set mode for filling points outside the input boundaries
    #     fill_mode='nearest',
    #     # value used for fill_mode = "constant"
    #     cval=0.,
    #     # randomly flip images
    #     horizontal_flip=True,
    #     # randomly flip images
    #     vertical_flip=False,
    #     # set rescaling factor (applied before any other transformation)
    #     rescale=None,
    #     # set function that will be applied on each input
    #     preprocessing_function=None,
    #     # image data format, either "channels_first" or "channels_last"
    #     data_format=None,
    #     # fraction of images reserved for validation (strictly between 0 and 1)
    #     validation_split=0.0)

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    train_flow = datagen.flow(x_train, y_train, batch_size=batch_size)
    # Fit the model on the batches generated by datagen.flow().
    if pa4val != 0:
        hist = model.fit_generator(
            train_flow,
            steps_per_epoch=train_flow.n // batch_size,
            validation_data=(x_valid, y_valid),
            validation_steps=x_valid.shape[0] // batch_size,
            epochs=epochs, verbose=1, workers=4,
            callbacks=callbacks
        )
    else:
        hist = model.fit_generator(
            train_flow,
            steps_per_epoch=train_flow.n // batch_size,
            epochs=epochs, verbose=1, workers=4,
            callbacks=callbacks
        )

if pa4val != 0:
    # Score trained model.
    scores = model.evaluate(x_valid, y_valid, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
else: pass

acc_train = hist.history['acc']
loss_train = hist.history['loss']
np.savetxt(os.path.join(project, 'acc.txt'), acc_train)
np.savetxt(os.path.join(project, 'loss.txt'), loss_train)

if pa4val != 0:
    acc_valid = hist.history['val_acc']
    loss_valid = hist.history['val_loss']
    np.savetxt(os.path.join(project, 'val_acc.txt'), acc_valid)
    np.savetxt(os.path.join(project, 'val_loss.txt'), loss_valid)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(range(epochs), acc_train, label='Train')
if pa4val != 0:
    ax.plot(range(epochs), acc_valid, label='Validate')
ax.legend()
plt.savefig(os.path.join(project, 'training_process.png'), dpi=300)
plt.show(block=False)
plt.close('all')

# In[]
train_record = pd.read_csv(r".\hw3\logs\D20191001T2143\training_logs_epoch000to199.csv")
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(train_record["epoch"], train_record["acc"], label='Train')
ax.plot(train_record["epoch"], train_record["val_acc"], label='Validate')
ax.legend()
plt.savefig(os.path.join(r".\hw3\logs\D20191001T2143\training_process.png"), dpi=300)
plt.show()

# In[]
from keras.models import load_model

model = load_model(r".\hw3\logs\D20191001T2143\hw3_ResNet50v2_epoch077.h5")
y_pred = model.predict(x_valid)

# In[]
from sklearn.metrics import confusion_matrix
y_target = np.argmax(y_valid, axis=1)
y_pred = np.argmax(y_pred, axis=1)

confu_mat = confusion_matrix(y_target, y_pred)
sum_confu_mat = np.sum(confu_mat, axis=1)
norm_confu_mat = confu_mat / sum_confu_mat
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.imshow(norm_confu_mat, cmap="jet")
plt.show()