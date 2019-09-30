# -*- coding: utf-8 -*-
"""
Created on 2019/9/29 下午 05:01
@author: Ivan Y.W.Chiu
"""
from keras import backend as K
from keras.models import Sequential, Model, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPool2D, AvgPool2D
from keras.layers import Flatten, LeakyReLU, UpSampling2D, Reshape, Input
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from keras.initializers import Constant, truncated_normal
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler, ReduceLROnPlateau
import os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
ROOT_PATH = os.getcwd()
sys.path.append(ROOT_PATH)


class NNModel():
    def __init__(self, mode, config, logdir=None, resume=None):
        self.mode = mode
        self.config = config
        self.logdir = logdir
        self.resume = resume
        self.checkpoint_name = '%s_epoch{epoch:03d}.h5' % config.NAME
        self.init_epoch = 0
        self.total_epoch = 0
        self.keras_model = self.build(mode, config)

    def build(self, mode, config):
        def conv2d(x, depth, kernel_size=config.CONV2D_KERNEL_SIZE,
                   stride=config.CONV2D_STRIDES, padding=config.CONV2D_PADDING,
                   kernel_init=config.KERNEL_INIT_METHOD,
                   bias_init=config.BIAS_INIT_DEFAULT):
            output = Conv2D(depth, kernel_size=kernel_size, strides=stride, padding=padding,
                            kernel_initializer=kernel_init,
                            bias_initializer=Constant(bias_init))(x)
            return output

        def activation(x, fn=config.ACTIVATION_FUNC):
            if fn == 'relu':
                output = Activation("relu")(x)
            elif fn == 'softmax':
                output = Activation("softmax")(x)
            elif fn == 'sigmoid':
                output = Activation("sigmoid")(x)
            elif fn == 'LeakyReLU':
                output = LeakyReLU(0.02)(x)
            else:
                output = x
            return output

        def maxpool2d(x, kernel_size=config.MAXPOOL2D_KERNEL_SIZE,
                      stride=config.MAXPOOL2D_STRIDES, padding=config.MAXPOOL2D_PADDING):
            output = MaxPool2D(pool_size=kernel_size, strides=stride, padding=padding)(x)
            return output

        def dense(x, units, kernel_init=config.KERNEL_INIT_METHOD,
                  bias_init=config.BIAS_INIT_DEFAULT):
            output = Dense(units, kernel_initializer=kernel_init, bias_initializer=Constant(bias_init))(x)
            return output

        def dropout(x, rate=config.DROPOUT_RATE):
            output = Dropout(rate)(x)
            return output

        K.clear_session()
        inputs = Input(shape=config.IMG_SHAPE)
        net = conv2d(inputs, 64)
        net = BatchNormalization()(net)
        net = activation(net)
        net = maxpool2d(net)
        net = conv2d(net, 128)
        net = BatchNormalization()(net)
        net = activation(net)
        net = maxpool2d(net)
        net = conv2d(net, 256)
        net = BatchNormalization()(net)
        net = activation(net)
        net = maxpool2d(net)
        net = Flatten()(net)
        net = dense(net, 512)
        net = BatchNormalization()(net)
        net = activation(net)
        net = dropout(net)
        net = dense(net, config.N_CLASS)
        net_out = activation(net, "softmax")
        model = Model(inputs, net_out)
        # model.summary()
        # create project
        model_summary(model)
        if self.mode == "training":
            if self.resume == 0:
                tn = time.localtime()
                self.project = self.logdir + "/D{0:4d}{1:02d}{2:02d}T{3:02d}{4:02d}".format(tn[0], tn[1], tn[2], tn[3], tn[4])
                os.mkdir(self.project)
        return model

    def train(self, training_set, validation_set, augmentation=0):
        # Prepare callbacks for model saving and for learning rate adjustment.
        checkpoint = ModelCheckpoint(filepath=os.path.join(self.project, self.checkpoint_name),
                                     monitor='val_acc',
                                     verbose=1)

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

        lr_scheduler = LearningRateScheduler(lr_schedule)

        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                       cooldown=0,
                                       patience=5,
                                       min_lr=0.5e-6)

        tensorboard = TensorBoard(log_dir=self.project, histogram_freq=0, write_graph=True, write_images=False)

        callbacks = [checkpoint, lr_reducer, lr_scheduler, tensorboard]

        self.keras_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.config.LR), metrics=['accuracy'])
        self.total_epoch = self.config.EPOCHS + self.init_epoch
        if augmentation == 0:
            print('Not using data augmentation.')
            hist = self.keras_model.fit(
                training_set.x_data, training_set.y_data,
                batch_size=self.config.BATCH_SIZE,
                epochs=self.config.EPOCHS + self.init_epoch,
                initial_epoch=self.init_epoch,
                validation_data=(validation_set.x_data, validation_set.y_data),
                shuffle=True,
                callbacks=callbacks
            )
        else:
            print('Using real-time data augmentation.')
            # This will do preprocessing and realtime data augmentation:
            datagen = ImageDataGenerator(
                    # set input mean to 0 over the dataset
                    featurewise_center = False,
                    # set each sample mean to 0
                    samplewise_center = False,
                    # divide inputs by std of dataset
                    featurewise_std_normalization = False,
                    # divide each input by its std
                    samplewise_std_normalization = False,
                    # apply ZCA whitening
                    zca_whitening = False,
                    # epsilon for ZCA whitening
                    zca_epsilon = 1e-06,
                    # randomly rotate images in the range (deg 0 to 180)
                    rotation_range = 0,
                    # randomly shift images horizontally
                    width_shift_range = 0.1,
                    # randomly shift images vertically
                    height_shift_range = 0.1,
                    # set range for random shear
                    shear_range = 0.,
                    # set range for random zoom
                    zoom_range = 0.,
                    # set range for random channel shifts
                    channel_shift_range = 0.,
                    # set mode for filling points outside the input boundaries
                    fill_mode = 'nearest',
                    # value used for fill_mode = "constant"
                    cval = 0.,
                    # randomly flip images
                    horizontal_flip = True,
                    # randomly flip images
                    vertical_flip = False,
                    # set rescaling factor (applied before any other transformation)
                    rescale = None,
                    # set function that will be applied on each input
                    preprocessing_function = None,
                    # image data format, either "channels_first" or "channels_last"
                    data_format = None,
                    # fraction of images reserved for validation (strictly between 0 and 1)
                    validation_split = 0.0)
            datagen.fit(training_set.x_data)
            train_flow = datagen.flow(training_set.x_data, training_set.y_data, batch_size=self.config.BATCH_SIZE)

            hist = self.keras_model.fit_generator(
                train_flow,
                steps_per_epoch=train_flow.n // self.config.BATCH_SIZE,
                epochs=self.total_epoch,
                initial_epoch=self.init_epoch,
                validation_data=(validation_set.x_data, validation_set.y_data),
                validation_steps=validation_set.x_data.shape[0] // self.config.BATCH_SIZE,
                shuffle=True,
                callbacks=callbacks
            )

        return hist

    def load_weights(self, file_path):
        self.model_name = os.path.basename(file_path).split(".")[0]
        K.clear_session()
        self.keras_model = load_model(file_path)
        if self.mode == "classify":
            self.project = file_path.split(os.path.basename(file_path))[0]
            print("Load weights " + os.path.basename(file_path) + " from " + self.project)
        else:
            if self.resume == 1:
                self.project = file_path.split(os.path.basename(file_path))[0]

    def get_init_epoch(self, resume_md_path):
        """ Get the initial epoch from given model directory
        resume_md_path (str): './path/to/model/model_name_epoch{epoch:03d}.h5'
        """
        resume_md_name = os.path.basename(resume_md_path)
        init_epoch = resume_md_name.split('_epoch')[-1].split('.')[0]
        self.init_epoch = int(init_epoch)

    def classify(self, images):
        predict_prob = self.keras_model.predict(images, verbose=1)
        predict_id = np.argmax(predict_prob, axis=1)
        out_df = pd.DataFrame({'id': list(range(predict_id.shape[0])), 'label': predict_id})
        out_df.to_csv(os.path.join(self.project, "classify_" + self.model_name + ".csv"), index=False)
        print("Saved prediction to " + os.path.abspath(os.path.join(self.project, "classify_" + self.model_name + ".csv")))
        return out_df

    def evaluate(self, prediction, labels):
        matching = np.where(prediction == labels, 1, 0)
        matching = matching.tolist()
        accuracy = matching.count(1) / len(matching)
        print('Test accuracy:', accuracy)
        return accuracy


def model_summary(keras_model, param_dict=None, valid_acc_dict=None, print_out=True, save_dir=None):
    str_list = []
    s = '{0:25s} | {1:25s} | {2:20s} | {3:8s} | {4:8s} | {5:20s}'.format(
        'Name', 'Input_shape', 'Kernel size', 'Strides', 'Padding', 'Output_shape'
    )
    str_list.append(s)
    str_list.append("-"*121)
    for l in keras_model.layers:
        if l.name.split('_')[0] == 'conv2d':
            s = '{0:25s} | {1:25s} | {2:20s} | {3:8s} | {4:8s} | {5:20s}'.format(
                l.name, str(l.input_shape), str(l.kernel._keras_shape), str(l.strides), l.padding, str(l.output_shape)
            )
            str_list.append(s)
        elif l.name.split('_')[0] == 'conv2d' and l.name.split('_')[1] == 'transpose':
            s = '{0:25s} | {1:25s} | {2:20s} | {3:8s} | {4:8s} | {5:20s}'.format(
                l.name, str(l.input_shape), str(l.kernel._keras_shape), str(l.strides), l.padding, str(l.output_shape)
            )
            str_list.append(s)

        elif l.name.split('_')[0] == 'batch':
            s = '{0:25s} | {1:25s} | {2:20s} | {3:8s} | {4:8s} | {5:20s}'.format(
                l.name, str(l.input_shape), "", "", "", ""
            )
            str_list.append(s)
        elif l.name.split('_')[0] == 'activation':
            str0 = l.output.name.split("/")
            s = '{0:25s} | {1:25s} | {2:20s} | {3:8s} | {4:8s} | {5:20s}'.format(
                str0[0]+"/"+str0[-1], "", "", "", "", "")
            str_list.append(s)
        elif l.name.split('_')[0] == 'leaky':
            s = '{0:25s}'.format(
                l.name)
            str_list.append(s)
        elif (l.name.split('_')[0] == 'max') and (l.name.split('_')[1] == 'pooling2d'):
            s = '{0:25s} | {1:25s} | {2:20s} | {3:8s} | {4:8s} | {5:20s}'.format(
                l.name, str(l.input_shape), str(l.pool_size), str(l.strides), l.padding, str(l.output_shape)
            )
            str_list.append(s)
        elif (l.name.split('_')[0] == 'avg') and (l.name.split('_')[1] == 'pooling2d'):
            s = '{0:25s} | {1:25s} | {2:20s} | {3:8s} | {4:8s} | {5:20s}'.format(
                l.name, str(l.input_shape), str(l.pool_size), str(l.strides), l.padding, str(l.output_shape)
            )
            str_list.append(s)
        elif (l.name.split('_')[0] == 'up') and (l.name.split('_')[1] == 'sampling2d'):
            s = '{0:25s} | {1:25s} | {2:20s} | {3:8s} | {4:8s} | {5:20s}'.format(
                l.name, str(l.input_shape), "", str(l.size), "", str(l.output_shape)
            )
            str_list.append(s)
        elif l.name.split('_')[0] == 'flatten' or l.name.split('_')[0] == 'reshape':
            s = '{0:25s} | {1:25s} | {2:20s} | {3:8s} | {4:8s} | {5:20s}'.format(
                l.name, str(l.input_shape), "", "", "", str(l.output_shape))
            str_list.append(s)
        elif l.name.split('_')[0] == 'dense':
            s = '{0:25s} | {1:25s} | {2:20s} | {3:8s} | {4:8s} | {5:20s}'.format(
                l.name, str(l.input_shape), str(l.units), "", "", str(l.output_shape)
            )
            str_list.append(s)
        elif l.name.split('_')[0] == 'dropout':
            s = '{0:25s} | {1:25s} | {2:20s} | {3:8s} | {4:8s} | {5:20s}'.format(l.name, str(l.rate), "", "", "", "")
            str_list.append(s)
    str_list.append("-"*121)

    if param_dict:
        for key in sorted(param_dict.keys()):
            s = key + ": " + str(param_dict[key])
            str_list.append(s)
        str_list.append("-" * 121)

    if valid_acc_dict:
        str_list.append(str(len(valid_acc_dict)) + "-FOLD VALIDATION ACCURACY")
        acc = []
        for key in sorted(valid_acc_dict.keys()):
            acc.append(valid_acc_dict[key])
            s = key + ": " + "{0:7.4f}%".format(100*valid_acc_dict[key])
            str_list.append(s)
        mean_acc = np.mean(np.asarray(acc))
        str_list.append("AVG: {0:7.4f}%".format(100*mean_acc))

    if print_out:
        for s in str_list:
            print(s)
    if save_dir:
        with open(save_dir, 'w') as f:
            for s in str_list:
                f.write(s + "\n")

    return str_list

