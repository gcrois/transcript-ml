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

import pickle

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

pickle.dump((X_train, X_test, y_train, y_test), open( "data/training_data.pickle", "wb" ))