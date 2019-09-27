# -*- coding: utf-8 -*-
"""
Homework 7 - Unsupervised Learning (Deep Autoencoder)
Created on 2019/7/19 下午 09:00
@author: Ivan Y.W.Chiu
"""

import sys, os, time
import numpy as np
import pandas as pd
import cv2
import src.tf_tools as nn
import src.keras_tools as knt
import src.keras_models as kmd
from keras.models import Sequential, Model, load_model
from keras.optimizers import SGD, Adam
from keras_preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
# from MulticoreTSNE import MulticoreTSNE as TSNE

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

#########################################################
# Data Processing
#########################################################
curr_dir = os.getcwd()
image_path = "./data/ml2019spring-hw7/images"
img_list = os.listdir(image_path)
os.chdir(image_path)
images = []
for im in img_list:
    img = cv2.imread(im)
    print("Loading image "+im, end="\r")
    images.append(img)

os.chdir(curr_dir)
images = np.array(images, dtype=np.float32)  # 32 x 32 x 3
images = images/255

#########################################################
# Build Model
#########################################################
epoch = 100
latent_dim = 256

K.clear_session()
tn = time.localtime()
project = "./logs/hw3/D{0:4d}{1:02d}{2:02d}T{3:02d}{4:02d}".format(tn[0], tn[1], tn[2], tn[3], tn[4])
os.mkdir(project)
auto_encoder, encoder, decoder = kmd.k_autoencoder3(latent_dim, project)
auto_encoder.compile(loss="MSE", optimizer=Adam(lr=0.0001))
mc = ModelCheckpoint(os.path.join(project, 'best_model.h5'), monitor='val_loss', mode='min',
                     verbose=1, save_best_only=True)
hist = auto_encoder.fit(images, images, batch_size=256, epochs=epoch, callbacks=[mc], validation_split=0.2)

loss_train = hist.history['loss']
loss_valid = hist.history['val_loss']

#########################################################
# Load trained model
#########################################################
# project = "./logs/hw7/D20190722T1545"
# latent_dim = 256
# auto_encoder = load_model(project + "/best_model.h5")
# # inputs = knt.md_input((32, 32, 3))
# inputs = auto_encoder.input
# encoding = auto_encoder.layers[1](inputs)
# for layer in auto_encoder.layers[2:24]:
#     encoding = layer(encoding)
# encoder = Model(inputs, encoding)
#
# encoded_input = knt.md_input(shape=(latent_dim,))
# decoding = auto_encoder.layers[24](encoded_input)
# for layer in auto_encoder.layers[25:]:
#     decoding = layer(decoding)
# decoder = Model(encoded_input, decoding)

#########################################################
# Encode & Decode
#########################################################
n2see = 10
x_test = images[-n2see:]
encoded = encoder.predict(x_test)
y_pred = decoder.predict(encoded)

fig = plt.figure("AutoEncoder", (20, 8))
for f in range(n2see):
    ax1 = fig.add_subplot(2, n2see, f + 1)
    ax1.imshow(x_test[f])
    ax1.set_axis_off()
    ax2 = fig.add_subplot(2, n2see, f + 1 + n2see)
    ax2.imshow(y_pred[f])
    ax2.set_axis_off()

plt.tight_layout()
plt.show(block=False)
fig.savefig(project + "/validation.png", dpi=300)
plt.close("all")

fig2 = plt.figure()
plt.rcParams.update({'font.size': 14})
ax3 = fig2.add_subplot(1, 1, 1)
ax3.plot(range(epoch), loss_train, label="Training")
ax3.plot(range(epoch), loss_valid, label="Validation")
ax3.legend()
ax3.set_xlabel("epoch")
ax3.set_ylabel("MSE")
ax3.set_title("Minimum validated MSE: " + "{0:.5f}".format(min(loss_valid)))
plt.tight_layout()
plt.show(block=False)
fig2.savefig(project + "/training_curve.png", dpi=300)
plt.close("all")

img_code = encoder.predict(images, batch_size=256)
np.savetxt(project + "/encode.txt", img_code)
# img_code = np.loadtxt(folder + "/encode.txt")

pca = PCA(n_components=149, copy=False, whiten=True, svd_solver="full")
pca_code = pca.fit_transform(img_code)

# tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=5000)
# tsne_code = tsne.fit_transform(pca_code)
# tsne = TSNE(n_jobs=6, n_components=2, verbose=1, perplexity=50, n_iter=5000)
# tsne_code = tsne.fit_transform(pca_code)
# plt.figure()
# plt.plot(tsne_code[:, 0], tsne_code[:, 1], '.')
# plt.plot(pca_code[:, 0], pca_code[:, 1], '.')

kmeans = KMeans(n_clusters=2, random_state=0).fit(pca_code)
cluster = kmeans.labels_

df = pd.read_csv("./data/ml2019spring-hw7/test_case.csv")
img1_id = df["image1_name"].values - 1
img2_id = df["image2_name"].values - 1
clust1 = cluster[img1_id]
clust2 = cluster[img2_id]
results = np.where(clust1 == clust2, 1, 0)
results_dict = {'id': list(range(clust1.shape[0])), 'label': results}
results_df = pd.DataFrame(results_dict)
results_df.to_csv(project + "/results_pca149.csv", index=False)
