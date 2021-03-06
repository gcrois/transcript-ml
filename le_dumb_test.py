#!/usr/bin/env python
# coding: utf-8

# # Residual CNN 
# 
# Tutorial used: https://adventuresinmachinelearning.com/introduction-resnet-tensorflow-2/


from definitions import *

import os
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
import pandas as pd

import pickle

#   Defines our 'block' of resnet   #
def res_net_block(input_data, filters, conv_size):
  x = layers.Conv2D(filters, conv_size, activation='relu', padding='same')(input_data)
  x = layers.BatchNormalization()(x)
  x = layers.Conv2D(filters, conv_size, activation=None, padding='same')(x)
  x = layers.BatchNormalization()(x)
  x = layers.Add()([x, input_data])
  x = layers.Activation('relu')(x)
  return x

X_train, X_test, y_train, y_test = pickle.load( open( "data/training_data.pickle", "rb" ) )

############### Begin making the model ###############################

print("Creating the 50 deep blocks")

# Make the tensor with matching dimensions #
inputs = keras.Input(shape=(70, 70, 3))

x = layers.Conv2D(32, 3, activation='relu')(inputs)
x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.MaxPooling2D(3)(x)

#   Loop through making our blocks  #
num_res_net_blocks = 50
for i in range(num_res_net_blocks):
    x = res_net_block(x, 64, 3)

x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)

outputs = layers.Dense(24, activation='softmax')(x)

res_net_model = keras.Model(inputs, outputs)
print("50 blocks done. Took:", round(time.time(), 2), " seconds")

############### Train ###############################

callbacks = [
  # Write TensorBoard logs to `./logs` directory
  keras.callbacks.TensorBoard(log_dir='./log/{}'.format(dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")), write_images=True),
 CSVLogger("model_history_log.csv", append=False)
]
res_net_model.compile(optimizer=keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])
history = res_net_model.fit(x=X_train, y=y_train, batch_size=138, epochs=75,
	  validation_data=(X_test, y_test), callbacks=callbacks)

data = pd.DataFrame(
    columns=["HiddenLayers", "LearningRate", "Optimizer",
            "NumFilters", "Activation", "KernelSize",
            "Momentum", "Epochs", "Loss",
            "Acc", "Val_Loss", "Val_Acc",]
)

for i in range(len(history.history["loss"])):
    data.loc[len(data)] = [
        "100",
        ".001",
        "Adam",
        "32,64,64,256",
        "relu",
        "3",
        "N/A",
        i, # epochs
        history.history["loss"][i],
        history.history["acc"][i],
        history.history["val_loss"][i],
        history.history["val_acc"][i],
    ]

data.to_csv(f"data/jS_stupid_results.csv")