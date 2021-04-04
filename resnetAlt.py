"""
    This code is a copy of the original resnet.py file but
    with changes made to the digital architecture of the
    model. This model is an attempt to follow the model
    as defined in its original 2015 paper as well as
    explore other, predefined resnet models.
"""


from definitions import *
from tensorflow import keras
from tensorflow.keras import layers
import time
import pandas as pd
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


def ResNetAlt(Epochs):
  # Load in all of our data #
  X_train, X_test, y_train, y_test = pickle.load(open("data/training_data.pickle", "rb"))

  ############### Begin making the model ###############################

  print("Creating the X deep blocks")
  tic = time.time()

  # Make the tensor with matching dimensions #
  inputs = keras.Input(shape=(70, 70, CHANNELS))

  x = layers.Conv2D(3, 3, activation="relu")(inputs)
  #x = layers.Conv2D(NumFilters, KernelSize, activation=Activation)(x)
  x = layers.MaxPooling2D(3)(x)

  #   Loop through making our blocks  #
  for i in range(20):
      x = res_net_block(x, 3, 3, "relu")

  # Pool, dense layer and ddropout  #
  x = layers.GlobalAveragePooling2D()(x)
  #x = layers.Dense(256, activation= "relu")(x)
  #x = layers.Dropout(0.5)(x)

  # Was originally softmax, might be for a good reason and should stay that way #
  outputs = layers.Dense(24, activation="softmax")(x)

  res_net_model = keras.Model(inputs, outputs)
  print("X blocks done. Took:", round(time.time() - tic, 2), " seconds")

  ############### Train ###############################

  res_net_model.compile(optimizer="adam",
                loss='sparse_categorical_crossentropy',
                metrics=['acc'])

  print(locals())
  history = res_net_model.fit(x=X_train, y=y_train, batch_size=128, epochs=Epochs,
      validation_data=(X_test, y_test), verbose=1)

  data = pd.DataFrame(
    columns=["Epochs", "Loss", "Acc", "Val_Loss", "Val_Acc"]
  )

  for i in range(len(history.history["loss"])):
    data.loc[len(data)] = [
      i + 1, # epochs
      history.history["loss"][i],
      history.history["acc"][i],
      history.history["val_loss"][i],
      history.history["val_acc"][i],
    ]

  return data
