#!/usr/bin/env python
# coding: utf-8

# # Residual CNN 
# 
# Tutorial used: https://adventuresinmachinelearning.com/introduction-resnet-tensorflow-2/


from definitions import *
from callback import *

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
from sklearn.model_selection import cross_val_score

import pickle

#   Defines our 'block' of resnet   #
#   Creates 2 blocks of resNet      #
def res_net_block(input_data, filters, conv_size, Activation):
  x = layers.Conv2D(filters, conv_size, activation=Activation, padding='same')(input_data)
  x = layers.BatchNormalization()(x)
  x = layers.Conv2D(filters, conv_size, activation=Activation, padding='same')(x)
  x = layers.BatchNormalization()(x)
  x = layers.Add()([x, input_data])
  x = layers.Activation(Activation)(x)
  return x


def ResNet(HiddenLayers, LearningRate, Optimizer, NumFilters, Activation, KernelSize, Momentum, Epochs, BatchSize, JobNum):
  # Load in all of our data #
  X, Y = pickle.load( open( "data/400k_training_data.pickle", "rb" ) )

  ############### Begin making the model ###############################

  print(f"Creating the {HiddenLayers} deep blocks")
  tic = time.time()

  # Make the tensor with matching dimensions #
  inputs = keras.Input(shape=(70, 70, CHANNELS))

  x = layers.Conv2D(NumFilters, KernelSize, activation=Activation)(inputs)
  x = layers.Conv2D(NumFilters, KernelSize, activation=Activation)(x)
  x = layers.MaxPooling2D(3)(x)

  #   Loop through making our blocks  #
  for i in range(HiddenLayers // 2):
      x = res_net_block(x, NumFilters, KernelSize, Activation)

  # Pool, dense layer and ddropout  #
  x = layers.GlobalAveragePooling2D()(x)
  x = layers.Dense(256, activation= Activation)(x)
  x = layers.Dropout(0.5)(x)

  # Was originally softmax, might be for a good reason and should stay that way #
  outputs = layers.Dense(24, activation="softmax")(x)

  res_net_model = keras.Model(inputs, outputs)
  print(f"{HiddenLayers} blocks done. Took:", round(time.time() - tic, 2), " seconds")

  ############### Train ###############################
  options = {opt: OPTMZ_ARGS[Optimizer][opt] for opt in OPTMZ_ARGS[Optimizer]}
  for o in options:
    options[o] = eval(options[o])

  res_net_model.compile(optimizer=eval(f"keras.optimizers.{Optimizer}")(**options),
                loss='sparse_categorical_crossentropy',
                metrics=['acc'])

  print(locals())
  scores = cross_val_score(res_net_model, X, Y, cv=10, scoring='accuracy')
  print(scores)
"""
  data = pd.DataFrame(
    columns=["HiddenLayers", "LearningRate", "Optimizer",
            "NumFilters", "Activation", "KernelSize",
            "Momentum", "Epochs", "Loss",
            "Acc", "Val_Loss", "Val_Acc",]
  )

  for i in range(len(history.history["loss"])):
    data.loc[len(data)] = [
      HiddenLayers,
      LearningRate,
      Optimizer,
      NumFilters,
      Activation,
      KernelSize,
      Momentum,
      i + 1, # epochs
      history.history["loss"][i],
      history.history["acc"][i],
      history.history["val_loss"][i],
      history.history["val_acc"][i],
    ]
"""
#  return scores
