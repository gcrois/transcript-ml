#   Generates the data for the 400k dataset #

from definitions import *

import os
import numpy as np
import cv2
from sklearn.utils import shuffle
import time

import pickle


# Read in the image #
def read_image(file_path):
    #   Reads image in grayscale    #
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
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
        data[i, :] = read_image(images + '/' + filename)
        y[i, 0] = classification[images.split("/")[-1]]
        i += 1

    return data, y


############### Run data processing ###############################
myData = 'data/400k/sorted/'
xf = []
yf = []

print("starting data processing")
tic = time.time()

#   Loop through directories    #
for x in os.listdir(myData):
    print(x)
    tmp = myData + x

    if os.path.isfile(tmp):
        continue

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

pickle.dump((X, Y), open("data/400k_training_data.pickle", "wb"), protocol=4)
