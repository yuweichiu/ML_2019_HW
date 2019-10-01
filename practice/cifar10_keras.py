# -*- coding: utf-8 -*-
"""
Created on 2019/9/29 下午 05:17
@author: Ivan Y.W.Chiu

python ./practice/cifar10_keras.py train --logs=./practice/logs
python ./practice/cifar10_keras.py train --logs=./practice/logs --augmentation=1
python ./practice/cifar10_keras.py train --logs=./practice/logs --augmentation=1 --resume=1 --weights=./practice/logs/D20190929T2220/cifar10_test_epoch020.h5
python ./practice/cifar10_keras.py detect --logs=./practice/logs --weights=./practice/logs/D20191001T0153/cifar10_ResNet56v2_epoch003.h5

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
    # ResNetv2 depth (9n+2): 
    RESNET_DEPTH = 20

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
    VALIDATION_RATE = 0


############################################################
#  Datasets
############################################################
class Cifar10Dataset(utils.Dataset):
    def feed_config(self, f_config):
        self.config = f_config
        self.n_class = f_config.N_CLASS
        self.img_shape = f_config.IMG_SHAPE
        self.val_rate = f_config.VALIDATION_RATE
        if self.val_rate == 0:
            self.use_val = False

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


############################################################
#  Train
############################################################
def train(model, config, augmentation=0):
    dataset = Cifar10Dataset()
    dataset.feed_config(config)
    dataset.load(usage='train')
    dataset.prepare()
    model.train(dataset, augmentation)


############################################################
#  Detection
############################################################
def detect(model, config):
    dataset = Cifar10Dataset()
    dataset.feed_config(config)
    dataset.load('detect')
    dataset.prepare()
    prediction = model.detect(dataset.x_data)
    accuracy = model.evaluate(prediction['label'].values, np.argmax(dataset.y_data, axis=1))


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
        detect(model, config)

