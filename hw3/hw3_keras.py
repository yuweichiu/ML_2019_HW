# -*- coding: utf-8 -*-
"""
Created on 2019/9/29 下午 05:17
@author: Ivan Y.W.Chiu

python ./hw3/hw3_keras.py train --logs=./hw3/logs
python ./hw3/hw3_keras.py train --logs=./hw3/logs --augmentation=1
python ./hw3/hw3_keras.py train --logs=./hw3/logs --augmentation=1 --resume=1 --weights=./hw3/logs/D20190929T2220/cifar10_test_epoch020.h5
python ./hw3/hw3_keras.py detect --logs=./hw3/logs --weights=./hw3/logs/D20190930T0035/cifar10_test_epoch020.h5

"""
import os, sys, time
ROOT_PATH = os.getcwd()
sys.path.append(ROOT_PATH)

# from keras.datasets import cifar10
import hw3.src.model as modelib
import hw3.src.utils as utils
from hw3.src.config import Config
import numpy as np
import pandas as pd

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
class Hw3Config(Config):
    # ResNetv2 depth (12n+2): 
    RESNET_DEPTH = 50

    # The name of your work:
    NAME = 'hw3_ResNet{:s}v2'.format(str(RESNET_DEPTH))

    # Number of class in dataset:
    N_CLASS = 7

    # The shape of image in dataset:
    IMG_SHAPE = [48, 48, 1]

    # Learning rate:
    LR = 0.001

    # Epochs:
    EPOCHS = 200

    # Training batch size:
    BATCH_SIZE = 64

    # Validation rate:
    VALIDATION_RATE = 0.2


############################################################
#  Datasets
############################################################
class Hw3Dataset(utils.Dataset):
    def feed_config(self, config):
        self.config = config
        self.n_class = config.N_CLASS
        self.img_shape = config.IMG_SHAPE
        self.val_rate = config.VALIDATION_RATE

    def load(self, usage):
        """
        Load the data (image, label) from dataset directory.
        :param usage: 'train' or 'detect'
        :return:
        """
        if usage == 'train':
            self.x_data = pd.read_csv('./data/ml2019spring-hw3/train_data.csv', header=None)
            self.y_data = pd.read_csv('./data/ml2019spring-hw3/train_label.csv', header=None)
            self.x_data, self.y_data = self.x_data.values, self.y_data.values
        else:
            self.x_data = pd.read_csv('./data/ml2019spring-hw3/test_data.csv', header=None)
            self.x_data = self.x_data.values


############################################################
#  Train
############################################################
def train(model, config, augmentation=0, resume=0):
    dataset = Hw3Dataset()
    dataset.feed_config(config)
    dataset.load(usage='train')
    dataset.prepare()
    model.train(dataset, augmentation)


############################################################
#  Detection
############################################################
def detect(model, config):
    # Testing dataset:
    dataset = Hw3Dataset()
    dataset.feed_config(config)
    dataset.load(usage='detect')
    dataset.prepare()

    prediction = model.detect(dataset_test.x_data)

############################################################
#  Main
############################################################
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Train HW3 datasets'
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
                        default=0,
                        metavar="/path/to/weight.h5",
                        help="Weight file directory")
    parser.add_argument('--augmentation', required=False,
                        default=0,
                        metavar="Data augmentation or not",
                        help='Use augmentation or not')
    args = parser.parse_args()

    # Configurations
    if args.mode == "train":
        config = Hw3Config()
        config_list = config.display(print_out=False)
    else:
        class ClassifyConfig(Hw3Config):
            BATCH_SIZE = 1
        config = ClassifyConfig()
        config_list = config.display(print_out=False)

    if args.mode == "train":
        model = modelib.NNModel(mode="training", config=config, logdir=args.logs, resume=int(args.resume))

        if int(args.resume) == 1:
            model.load_weights(args.weights)
            model.get_init_epoch(args.weights)

        train(model, config, augmentation=int(args.augmentation), resume=int(args.resume))
    else:
        model = modelib.NNModel(mode="classify", config=config, logdir=args.logs, resume=0)
        model.load_weights(args.weights)
        detect(model, config)






