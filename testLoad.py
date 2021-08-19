#   Using this to create the confusion matrix   #
import tensorflow as tf
from ModelCreation import *
import numpy as np

#   Import the model    #
print("Creating the model")
model = ResNetModel(HiddenLayers=32, LearningRate=.001, Optimizer="RMSprop", NumFilters=96, Activation="relu", KernelSize=3, Momentum=.9, Epochs=1, BatchSize=512, JobNum=1)

#   Load the pretrained weights #
print("Loading the weights")
model.load_weights("training_1/cp.ckpt")

print(model.summary())

"""
print("Quick evaluation: ")
print(model.evaluate(X, Y))

print("Predicting")
predictions = model.predict(X)

matrix = tf.math.confusion_matrix(Y, predictions, num_classes=24)

print("Trying to print the matrix")
print(matrix)
"""
