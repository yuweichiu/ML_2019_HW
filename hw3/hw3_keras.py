# -*- coding: utf-8 -*-
"""
Created on 2019/9/29 下午 05:17
@author: Ivan Y.W.Chiu

python ./practice/hw3_keras.py train --logs=./practice/logs
python ./practice/hw3_keras.py train --logs=./practice/logs --augmentation=1
python ./practice/hw3_keras.py train --logs=./practice/logs --augmentation=1 --resume=1 --weights=./practice/logs/D20190929T2220/cifar10_test_epoch020.h5
python ./practice/hw3_keras.py classify --logs=./practice/logs --weights=./practice/logs/D20190930T0035/cifar10_test_epoch020.h5

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
    # The name of your work:
    NAME = 'cifar10_test'

    # Number of class in dataset:
    N_CLASS = 10

    # The shape of image in dataset:
    IMG_SHAPE = [32, 32, 3]

    # Learning rate:
    LR = 0.001

    # Epochs:
    EPOCHS = 20

    # Training batch size:
    BATCH_SIZE = 256

    # Validation rate:
    VALIDATION_RATE = 0.2


############################################################
#  Datasets
############################################################
class Cifar10Dataset(utils.Dataset):
    def feed_config(self, config):
        self.n_class = config.N_CLASS
        self.img_shape = config.IMG_SHAPE
        self.val_rate = config.VALIDATION_RATE

    def load(self, usage):
        if usage == 'train':
            (self.x_data, self.y_data), _ = cifar10.load_data()
            total = self.x_data.shape[0]
            self.x_data = self.x_data[0: int(total * (1 - self.val_rate))]
            self.y_data = self.y_data[0: int(total * (1 - self.val_rate))]
        elif usage == 'val':
            (self.x_data, self.y_data), _ = cifar10.load_data()
            total = self.x_data.shape[0]
            self.x_data = self.x_data[int(total * (1 - self.val_rate)): ]
            self.y_data = self.y_data[int(total * (1 - self.val_rate)): ]
        else:
            _, (self.x_data, self.y_data) = cifar10.load_data()


############################################################
#  Train
############################################################
def train(model, config, augmentation=0, resume=0):
    # Training dataset:
    dataset_train = Cifar10Dataset()
    dataset_train.feed_config(config)
    dataset_train.load('train')
    dataset_train.prepare()

    # Validation dataset:
    dataset_val = Cifar10Dataset()
    dataset_val.feed_config(config)
    dataset_val.load('val')
    dataset_val.prepare()

    model.train(dataset_train, dataset_val, augmentation)
    # model.save_training_log(model.train_hist, resume)
    # model.plot_training_hist(target='loss')
    # model.plot_training_hist(target='acc')


############################################################
#  Classification
############################################################
def classify(model, config):
    # Testing dataset:
    dataset_test = Cifar10Dataset()
    dataset_test.feed_config(config)
    dataset_test.load('test')
    dataset_test.prepare()

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
        config = Cifar10Config()
    else:
        class ClassifyConfig(Cifar10Config):
            BATCH_SIZE = 1
        config = ClassifyConfig()

    if args.mode == "train":
        model = modelib.NNModel(mode="training", config=config, logdir=args.logs, resume=int(args.resume))

        if int(args.resume) == 1:
            model.load_weights(args.weights)
            model.get_init_epoch(args.weights)

        train(model, config, augmentation=int(args.augmentation), resume=int(args.resume))
    else:
        model = modelib.NNModel(mode="classify", config=config, logdir=args.logs, resume=0)
        model.load_weights(args.weights)
        classify(model, config)






