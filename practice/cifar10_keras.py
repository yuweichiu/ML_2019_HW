# -*- coding: utf-8 -*-
"""
Created on 2019/9/29 下午 05:17
@author: Ivan Y.W.Chiu

python ./practice/cifar10_keras.py train --logs=./practice/logs
python ./practice/cifar10_keras.py train --logs=./practice/logs --augmentation=1
python ./practice/cifar10_keras.py train --logs=./practice/logs --augmentation=1 --resume=1 --weights=./practice/logs/D20190929T2220/cifar10_test_epoch020.h5
python ./practice/cifar10_keras.py classify --logs=./practice/logs --weights=./practice/logs/D20190930T0035/cifar10_test_epoch020.h5

"""
import os, sys, time
ROOT_PATH = os.getcwd()
sys.path.append(ROOT_PATH)

from keras.datasets import cifar10
# from practice.src import model as modellib
import practice.src.model as modelib
import practice.src.utils as utils
from practice.src.config import Config
import numpy as np

DEFAULT_LOG_PATH = os.path.join(ROOT_PATH, "logs")

############################################################
#  Session Setting
############################################################
# If you face the error about convolution layer,
# use this block to enable the memory usage of GPU growth.
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.Session(config=tf_config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

############################################################
#  Configurations
############################################################
class Cifar10Config(Config):
    # ResNetv2 depth:
    RESNET_DEPTH = 56

    # The name of your work:
    NAME = 'cifar10_ResNet{:s}v2'.format(str(RESNET_DEPTH))

    # Number of class in dataset:
    N_CLASS = 10

    # The shape of image in dataset:
    IMG_SHAPE = [32, 32, 3]

    # Learning rate:
    LR = 0.001

    # Epochs:
    EPOCHS = 3

    # Training batch size:
    BATCH_SIZE = 256

    # Validation rate:
    VALIDATION_RATE = 0.2


############################################################
#  Datasets
############################################################
class Cifar10Dataset(utils.Dataset):
    def feed_config(self, f_config):
        self.config = f_config
        self.n_class = f_config.N_CLASS
        self.img_shape = f_config.IMG_SHAPE
        self.val_rate = f_config.VALIDATION_RATE

    def load(self, usage):
        """
        Load the data (image, label) from dataset directory.
        :param usage: 'train' or 'detect'
        :return:
        """
        if usage == 'train':
            (self.x_data, self.y_data), _ = cifar10.load_data()
        else:
            _, (self.x_data, self.y_data) = cifar10.load_data()

    def split(self):
        """
        Split the dataset as training set or validation set
        :param usage (str): 'training_set' or 'validation_set'
        :return:
        """
        val_rate = self.config.VALIDATION_RATE
        total = self.x_data.shape[0]

        # For training set:
        self.x_train = self.x_data[0: int(total * (1 - val_rate))]
        self.y_train = self.y_data[0: int(total * (1 - val_rate))]
        print('Training set: {:d}'.format(int(total * (1 - val_rate))))

        # For validation set:
        # If validation rate is 0, means don't use validation set.
        if val_rate == 0:
            self.use_val = False
            print("Validation set: N/A")
            pass
        else:
            self.x_val = self.x_data[int(total * (1 - self.val_rate)): ]
            self.y_val = self.y_data[int(total * (1 - self.val_rate)): ]
            # self.validation_set = (self.x_val, self.y_val)
            print('Validation set: {:d}'.format(int(total * (1 - self.val_rate))))


    # def load(self, usage):
    #     if usage == 'train':
    #         (self.x_data, self.y_data), _ = cifar10.load_data()
    #         total = self.x_data.shape[0]
    #         self.x_data = self.x_data[0: int(total * (1 - self.val_rate))]
    #         self.y_data = self.y_data[0: int(total * (1 - self.val_rate))]
    #     elif usage == 'val':
    #         (self.x_data, self.y_data), _ = cifar10.load_data()
    #         total = self.x_data.shape[0]
    #         self.x_data = self.x_data[int(total * (1 - self.val_rate)): ]
    #         self.y_data = self.y_data[int(total * (1 - self.val_rate)): ]
    #     else:
    #         _, (self.x_data, self.y_data) = cifar10.load_data()


############################################################
#  Train
############################################################
def train(model, config, augmentation=0):
    dataset = Cifar10Dataset()
    dataset.feed_config(config)
    dataset.load(usage='train')
    dataset.prepare()
    # dataset.split()

    # # Training dataset:
    # dataset_train = Cifar10Dataset()
    # dataset_train.feed_config(config)
    # dataset_train.load('train')
    # dataset_train.prepare()
    #
    # # Validation dataset:
    # dataset_val = Cifar10Dataset()
    # dataset_val.feed_config(config)
    # dataset_val.load('val')
    # dataset_val.prepare()

    # dataset.x_data, dataset.x_train, dataset.x_val = dataset.x_data / 255, dataset.x_train / 255, dataset.x_val / 255
    # dataset_train.x_data, dataset_val.x_data = dataset_train.x_data / 255, dataset_val.x_data / 255
    # if config.SUBTRACT_PIXEL_MEAN is True:
    #     x_mean = np.mean(dataset.x_data, axis=0)
    #     dataset.x_train -= x_train_mean
    #     dataset_val.x_data -= x_train_mean

    model.train(dataset, augmentation)


############################################################
#  Classification
############################################################
def classify(model, config):
    # Training dataset:
    dataset_train = Cifar10Dataset()
    dataset_train.feed_config(config)
    dataset_train.load('train')
    dataset_train.prepare()

    # Testing dataset:
    dataset_test = Cifar10Dataset()
    dataset_test.feed_config(config)
    dataset_test.load('test')
    dataset_test.prepare()

    dataset_train.x_data, dataset_test.x_data = dataset_train.x_data / 255, dataset_test.x_data / 255
    if config.SUBTRACT_PIXEL_MEAN is True:
        x_train_mean = np.mean(dataset_train.x_data, axis=0)
        dataset_train.x_data -= x_train_mean
        dataset_test.x_data -= x_train_mean

    prediction = model.classify(dataset_test.x_data)
    accuracy = model.evaluate(prediction['label'].values, np.argmax(dataset_test.y_data, axis=1))


############################################################
#  Main
############################################################
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Train Cifar10'
    )
    parser.add_argument("mode",
                        metavar="<mode>",
                        help='"train" or "classify"')
    parser.add_argument("--logs", required=False,
                        default=DEFAULT_LOG_PATH,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument("--resume", required=False,
                        default=0,
                        metavar='0 or 1',
                        help="Resume training / use as initial weight")
    parser.add_argument("--weights", required=False,
                        default=None,
                        metavar="/path/to/weight.h5",
                        help="Weight file directory")
    parser.add_argument('--augmentation', required=False,
                        default=0,
                        metavar="Data augmentation or not",
                        help='Use augmentation or not')
    args = parser.parse_args()

    # Configurations
    if args.mode == "train":
        config = Cifar10Config()
        config_list = config.display(print_out=False)
    else:
        class ClassifyConfig(Cifar10Config):
            BATCH_SIZE = 1
        config = ClassifyConfig()
        config_list = config.display(print_out=False)

    if args.mode == "train":
        model = modelib.NNModel(mode="training", config=config, logdir=args.logs, resume=int(args.resume))

        if int(args.resume) == 1:
            # TODO: verify the resume mechanism
            model.load_weights(args.weights)
            model.get_init_epoch(args.weights)
        else:
            if args.weights:
                model.load_weights(args.weights)
            else:
                print("No weights specified, starting a new training work.")

        train(model, config, augmentation=int(args.augmentation))
    else:
        model = modelib.NNModel(mode="classify", config=config, logdir=args.logs, resume=0)
        model.load_weights(args.weights)
        classify(model, config)

