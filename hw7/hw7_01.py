# -*- coding: utf-8 -*-
"""
Homework 7 - Unsupervised Learning (Handcraft PCA)
Created on 2019/7/18 下午 06:42
@author: Ivan Y.W.Chiu
"""

import os
import sys
import numpy as np
from skimage.io import imread, imsave
# import cv2

IMAGE_PATH = r'./data/ml2019spring-hw7/Aberdeen'

# Images for compression & reconstruction
test_image = ['1.jpg', '10.jpg', '22.jpg', '37.jpg', '72.jpg']

# Number of principal components used
k = 5


def process(M):
    M -= np.min(M)
    M /= np.max(M)
    M = (M * 255).astype(np.uint8)
    return M


filelist = os.listdir(IMAGE_PATH)

# Record the shape of images
img_shape = imread(os.path.join(IMAGE_PATH, filelist[0])).shape

img_data = []
for filename in filelist:
    tmp = imread(os.path.join(IMAGE_PATH, filename))
    img_data.append(tmp.flatten())

training_data = np.array(img_data).astype('float32')

# Calculate mean & Normalize
mean = np.mean(training_data, axis=0)
training_data -= mean

# Use SVD to find the eigenvectors
# np.linalg.svd()
u, s, v = np.linalg.svd(training_data, full_matrices=False)
S = np.diag(s[0:k])
for x in test_image:
    # Load image & Normalize
    print(x)
    picked_img = imread(os.path.join(IMAGE_PATH, x))
    X = picked_img.flatten().astype('float32')
    X -= mean

    # Compression
    # weight = np.array([s.dot(v) for i in range(k)])
    weight = S.dot(v[0:5])
    #
    # Reconstruction
    i = int(filelist.index(x))
    # reconstruct = process(u[i, 0:k].dot(weight) + mean)
    reconstruct = process(u[i, 0:k].dot(weight) + mean)
    imsave(os.path.join("./hw7", "reconstruction_" + x[:-4] + '.jpg'), reconstruct.reshape(img_shape))

average = process(mean)
imsave(os.path.join("./hw7", 'average.jpg'), average.reshape(img_shape))

for x in range(5):
    eigenface = process(weight)
    imsave(os.path.join("./hw7", "eigenface_" + str(x) + '.jpg'), eigenface.reshape(img_shape))

for i in range(5):
    number = s[i] * 100 / sum(s)
    print(number)