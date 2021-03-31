#!/usr/bin/env python
# coding: utf-8

# # Residual CNN 
# 
# Tutorial used: https://adventuresinmachinelearning.com/introduction-resnet-tensorflow-2/


import os
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import datetime as dt
import itertools
import time
from tensorflow.keras.callbacks import CSVLogger

classification = {
    "Alpha" : 0,
    "Beta" : 1,
    "Chi" : 2,
    "Delta" : 3,
    "Epsilon" : 4,
    "Eta" : 5,
    "Gamma" : 6,
    "Iota" : 7,
    "Kappa" : 8,
    "Lambda" : 9,
    "Mu" : 10,
    "Nu" : 11,
    "Omega" : 12,
    "Omicron" : 13,
    "Phi" : 14,
    "Pi" : 15,
    "Psi" : 16,
    "Rho" : 17,
    "Sigma" : 18,
    "Tau" : 19,
    "Theta" : 20,
    "Upsilon" : 21,
    "Xi" :22,
    "Zeta" : 23
}


ROWS = 70
COLS = 70
CHANNELS = 3
CLASSES = 24
LENGTH = 1000

# Read in the image #
def read_image(file_path):

    #   Reads image in grayscale    #
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img = img[..., np.newaxis]
    return img

# For parsing through directory and loading array of images #
def prep_data(images):
    path, dirs, files = next(os.walk(images))
    length = len(files)
    
    # make an array for the images #
    data = np.zeros((length, ROWS, COLS, CHANNELS), dtype=np.uint8)
    
    # make an array for the identites #
    y = np.zeros((length, 1), dtype=np.uint8)
    
    i = 0
    for filename in os.listdir(images):

        #   Used for testing, uncomment to run through certain split of data #
#        if(i == LENGTH):
#            break
        data[i,:] = read_image(images + '/' + filename)
        y[i,0] = classification[images.split("/")[2]]
        i += 1
        
    return data, y

#   Defines our 'block' of resnet   #
def res_net_block(input_data, filters, conv_size):
  x = layers.Conv2D(filters, conv_size, activation='relu', padding='same')(input_data)
  x = layers.BatchNormalization()(x)
  x = layers.Conv2D(filters, conv_size, activation=None, padding='same')(x)
  x = layers.BatchNormalization()(x)
  x = layers.Add()([x, input_data])
  x = layers.Activation('relu')(x)
  return x


############### Run data processing ###############################
myData = '400k/sorted/'
xf = []
yf = []
tic = time.time()
print("starting data processing")

#   Loop through directories    #
for x in os.listdir(myData):
    tmp = myData + x
    x, y = prep_data(tmp)
    xf.append(x)
    yf.append(y)

print("data processing done. Took:", round(time.time() - tic, 2), " seconds")

#   Fix array dimensionality    #
combinedX = [item for sublist in xf for item in sublist]
yl = [item for sublist in yf for item in sublist]
X, Y = shuffle(combinedX, yl)
X = np.array(X)
Y = np.array(Y)

# Split the training and testing data #
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=1)


############### Begin making the model ###############################

print("Creating the 50 deep blocks")

# Make the tensor with matching dimensions #
inputs = keras.Input(shape=(70, 70, 1))

x = layers.Conv2D(32, 3, activation='relu')(inputs)
x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.MaxPooling2D(3)(x)

#   Loop through making our blocks  #
num_res_net_blocks = 10
for i in range(num_res_net_blocks):
    x = res_net_block(x, 64, 3)

x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)

outputs = layers.Dense(24, activation='softmax')(x)

res_net_model = keras.Model(inputs, outputs)
print("50 blocks done. Took:", round(time.time() - tic, 2), " seconds")


############### Train ###############################

callbacks = [
  # Write TensorBoard logs to `./logs` directory
  keras.callbacks.TensorBoard(log_dir='./log/{}'.format(dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")), write_images=True),
 CSVLogger("model_history_log.csv", append=False)
]
res_net_model.compile(optimizer=keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])
history = res_net_model.fit(x=X_train, y=y_train, batch_size=138, epochs=10,
	  validation_data=(X_test, y_test), callbacks=callbacks)


