# -*- coding: utf-8 -*-
"""
Created on 2019/9/29 下午 05:01
@author: Ivan Y.W.Chiu
"""
import keras
from keras import backend as K
from keras.models import Sequential, Model, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPool2D, AvgPool2D
from keras.layers import Flatten, LeakyReLU, UpSampling2D, Reshape, Input
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from keras.initializers import Constant, truncated_normal
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler, ReduceLROnPlateau, CSVLogger
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
        self.train_hist = None
        self.keras_model = self.build(config)

    def build(self, config):
        # model = self.CNN_simple(config)
        model = self.ResNetv2(config)

        # create project
        if self.mode == "training":
            if self.resume == 0:
                tn = time.localtime()
                self.project = self.logdir + "/D{0:4d}{1:02d}{2:02d}T{3:02d}{4:02d}".format(tn[0], tn[1], tn[2], tn[3], tn[4])
                os.mkdir(self.project)
        return model

    def CNN_simple(self, config):
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
        return model

    def ResNetv2(self, config):
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
        stage 2:  12x12,  256
        stage 3:  6 x 6,  512


        # Arguments
            input_shape (tensor): shape of input image tensor
            depth (int): number of core convolutional layers
            num_classes (int): number of classes (CIFAR10 has 10)

        # Returns
            model (Model): Keras model instance
        """

        def resnet_layer(inputs,
                         num_filters=16,
                         kernel_size=config.CONV2D_KERNEL_SIZE,
                         strides=config.CONV2D_STRIDES,
                         activation=config.ACTIVATION_FUNC,
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

        if (config.RESNET_DEPTH - 2) % 12 != 0:
            raise ValueError('depth should be 12n+2 (eg 50 or 110 in [b])')
        # Start model definition.
        num_filters_in = 16
        num_res_blocks = int((config.RESNET_DEPTH - 2) / 12)

        inputs = Input(shape=config.IMG_SHAPE)
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
                        strides = 2  # downsample

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
        x = AvgPool2D(pool_size=6)(x)
        y = Flatten()(x)
        outputs = Dense(config.N_CLASS,
                        activation='softmax',
                        kernel_initializer='he_normal')(y)

        # Instantiate model.
        model = Model(inputs=inputs, outputs=outputs)
        return model

    def train(self, dataset, augmentation=0):
        model_summary(self.keras_model, self.config.list, save_dir=os.path.join(self.project, "model_summary.txt"))

        # Normalize dataset and split the dataset according to validation set:
        dataset.x_data = dataset.x_data / 255
        dataset.split()
        if self.config.SUBTRACT_PIXEL_MEAN is True:
            x_mean = np.mean(dataset.x_data, axis=0)
            np.save(os.path.join(self.project, 'mean_img'), x_mean)
            dataset.x_train -= x_mean
            if dataset.use_val:
                dataset.x_val -= x_mean
        if dataset.use_val:
            dataset.validation_set = (dataset.x_val, dataset.y_val)
            validation_steps = dataset.x_val.shape[0] // self.config.BATCH_SIZE
        else:
            dataset.validation_set = None
            validation_steps = None

        if dataset.use_val is True:
            monitor = 'val_acc'
        else:
            monitor = 'acc'
        print("Using {:s} as the monitor.".format(monitor))

        # Prepare callbacks for model saving and for learning rate adjustment.
        checkpoint = ModelCheckpoint(filepath=os.path.join(self.project, self.checkpoint_name),
                                     monitor=monitor,
                                     verbose=1,
                                     save_best_only=self.config.SAVE_BEST_ONLY)

        csv_logger = CSVLogger(filename=os.path.join(self.project, 'training_logs_epoch%03dto%03d.csv' %
                                                     (self.init_epoch, self.init_epoch + self.config.EPOCHS - 1
        )))

        def lr_schedule(epoch):
            """Learning Rate Schedule

            Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
            Called automatically every epoch as part of callbacks during training.

            # Arguments
                epoch (int): The number of epochs

            # Returns
                lr (float32): learning rate
            """
            lr = self.config.LR
            if epoch > 150:
                lr *= 1e-1
            elif epoch > 110:
                lr *= 0.005
            elif epoch > 80:
                lr *= 0.01
            elif epoch > 50:
                lr *= 0.1
            print('Learning rate: ', lr)
            return lr

        lr_scheduler = LearningRateScheduler(lr_schedule)

        lr_reducer = ReduceLROnPlateau(monitor=monitor, factor=np.sqrt(0.1),
                                       cooldown=0, patience=5, min_lr=0.5e-6)

        tensorboard = TensorBoard(log_dir=self.project, histogram_freq=0, write_graph=True, write_images=False)

        callbacks = [checkpoint, lr_reducer, lr_scheduler, tensorboard, csv_logger]

        self.keras_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr_schedule(0)), metrics=['accuracy'])
        self.total_epoch = self.config.EPOCHS + self.init_epoch
        
        # Start training:
        if augmentation == 0:
            print('Not using data augmentation.')
            self.train_hist = self.keras_model.fit(
                dataset.x_train, dataset.y_train,
                batch_size=self.config.BATCH_SIZE,
                epochs=self.config.EPOCHS + self.init_epoch,
                initial_epoch=self.init_epoch,
                validation_data=dataset.validation_set,
                shuffle=True,
                callbacks=callbacks
            )
        else:
            print('Using real-time data augmentation.')
            # This will do preprocessing and realtime data augmentation:
            datagen = ImageDataGenerator(
                    # randomly rotate images in the range (deg 0 to 180)
                    rotation_range = 10,
                    # randomly shift images horizontally
                    width_shift_range = 0.2,
                    # randomly shift images vertically
                    height_shift_range = 0.2,
                    # set range for random shear
                    shear_range = 0.,
                    # set range for random zoom
                    zoom_range = 0.2,
                    # set mode for filling points outside the input boundaries
                    fill_mode = 'nearest',
                    # value used for fill_mode = "constant"
                    cval = 0.,
                    # randomly flip images
                    horizontal_flip = True,
                    # randomly flip images
                    vertical_flip = False)
            datagen.fit(dataset.x_train)
            train_flow = datagen.flow(dataset.x_train, dataset.y_train, batch_size=self.config.BATCH_SIZE)

            self.train_hist = self.keras_model.fit_generator(
                train_flow,
                steps_per_epoch=train_flow.n // self.config.BATCH_SIZE,
                epochs=self.total_epoch,
                initial_epoch=self.init_epoch,
                validation_data=dataset.validation_set,
                validation_steps=validation_steps,
                shuffle=True,
                callbacks=callbacks
            )

    def save_training_log(self, train_hist, resume):
        cur_acc_train = train_hist.history['acc']
        cur_loss_train = train_hist.history['loss']
        cur_acc_valid = train_hist.history['val_acc']
        cur_loss_valid = train_hist.history['val_loss']

        if resume == 1:
            old_acc_train = np.loadtxt(os.path.join(self.project, "acc.txt"))
            old_loss_train = np.loadtxt(os.path.join(self.project, "loss.txt"))
            old_acc_valid = np.loadtxt(os.path.join(self.project, "val_acc.txt"))
            old_loss_valid = np.loadtxt(os.path.join(self.project, "val_loss.txt"))
            self.acc_train = np.concatenate([old_acc_train, cur_acc_train])
            self.loss_train = np.concatenate([old_loss_train, cur_loss_train])
            self.acc_valid = np.concatenate([old_acc_valid, cur_acc_valid])
            self.loss_valid = np.concatenate([old_loss_valid, cur_loss_valid])
        else:
            self.acc_train = cur_acc_train
            self.loss_train = cur_loss_train
            self.acc_valid = cur_acc_valid
            self.loss_valid = cur_loss_valid

        np.savetxt(os.path.join(self.project, 'acc.txt'), self.acc_train)
        np.savetxt(os.path.join(self.project, 'loss.txt'), self.loss_train)
        np.savetxt(os.path.join(self.project, 'val_acc.txt'), self.acc_valid)
        np.savetxt(os.path.join(self.project, 'val_loss.txt'), self.loss_valid)

    def plot_training_hist(self, target='loss'):
        # log_df = pd.read_csv(os.path.join(self.project, "training_logs.csv"))
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        if target == 'loss':
            ax.plot(self.loss_train.shape[0], self.loss_train, label='Training')
            ax.plot(self.loss_train.shape[0], self.loss_valid, label='Validation')
            ax.set_title("Training Loss")
        else:
            ax.plot(self.acc_train.shape[0], self.acc_train, label='Training')
            ax.plot(self.acc_train.shape[0], self.acc_valid, label='Validation')
            ax.set_title("Training accuracy")
        ax.legend()
        plt.tight_layout()
        plt.show(block=False)
        plt.savefig(os.path.join(self.project, 'training_history_' + target + '.png'), dpi=300)
        # plt.close('all')

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

    def detect(self, images):
        images = images / 255
        if self.config.SUBTRACT_PIXEL_MEAN is True:
            x_mean = np.load(os.path.join(self.project, 'mean_img.npy'))
            images -= x_mean
        predict_prob = self.keras_model.predict(images, verbose=1)
        predict_id = np.argmax(predict_prob, axis=1)
        out_df = pd.DataFrame({'id': list(range(predict_id.shape[0])), 'label': predict_id})
        out_df.to_csv(os.path.join(self.project, "detect_" + self.model_name + ".csv"), index=False)
        print("Saved prediction to " + os.path.abspath(os.path.join(self.project, "detect_" + self.model_name + ".csv")))
        return out_df

    def evaluate(self, prediction, labels):
        matching = np.where(prediction == labels, 1, 0)
        matching = matching.tolist()
        accuracy = matching.count(1) / len(matching)
        print('Test accuracy:', accuracy)
        return accuracy


def model_summary(keras_model, config_list=None, print_out=True, save_dir=None):
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

    if config_list:
        str_list = str_list + config_list
        str_list.append("-" * 121)

    if print_out:
        for s in str_list:
            print(s)
    if save_dir:
        with open(save_dir, 'w') as f:
            for s in str_list:
                f.write(s + "\n")

    return str_list

