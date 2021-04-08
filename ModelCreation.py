#   Creates just the model framework    #

from definitions import *
from callback import *

import os
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import time
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


def ResNetModel(HiddenLayers, LearningRate, Optimizer, NumFilters, Activation, KernelSize, Momentum, Epochs, BatchSize, JobNum):
  # Load in all of our data #

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


  # Define the K-fold Cross Validator
  res_net_model.compile(optimizer=eval(f"keras.optimizers.{Optimizer}")(**options),
                        loss='sparse_categorical_crossentropy',
                        metrics=['acc'])

  return res_net_model
