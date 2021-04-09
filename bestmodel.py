#!/usr/bin/env python
# coding: utf-8

# # Residual CNN 
# 
# Tutorial used: https://adventuresinmachinelearning.com/introduction-resnet-tensorflow-2/


from definitions import *
from callback import *
from ModelCreation import *

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
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
import csv

import pandas as pd

import pickle

def ResNet(HiddenLayers, LearningRate, Optimizer, NumFilters, Activation, KernelSize, Momentum, Epochs, BatchSize, JobNum):

  X, Y = pickle.load( open( "data/400k_training_data.pickle", "rb"))

  X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1, random_state = 1)

  res_net_model = ResNetModel(HiddenLayers, LearningRate, Optimizer, NumFilters, Activation, KernelSize, Momentum, Epochs, BatchSize, JobNum)


  checkpoint_path = "training_1/cp.ckpt"
  checkpoint_dir = os.path.dirname(checkpoint_path)

  # Create a callback that saves the model's weights
  cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                   save_weights_only=True,
                                                   verbose=1)

  history = res_net_model.fit(x=X_train, y=y_train, batch_size=BatchSize, verbose=1, epochs=Epochs, callbacks=[cp_callback])
  res_net_model.evaluate(X_test, y_test)



  print("Running K Cross")

  """
  for train, valid in kfold.split(X, Y):


    

    print("Iteration ", spot, ": ")
    print('Model evaluation ', res_net_model.evaluate(X[valid],Y[valid]))
    with open('validData.csv', 'a') as csvfile:
      csvwriter = csv.writer(csvfile)
      csvwriter.writerow(res_net_model.evaluate(X[valid], Y[valid]))
    spot += 1

  """
  return res_net_model


ResNet(HiddenLayers=32, LearningRate=.001, Optimizer="RMSprop", NumFilters=96, Activation="relu", KernelSize=3, Momentum=.9, Epochs=50, BatchSize=512, JobNum=1)
