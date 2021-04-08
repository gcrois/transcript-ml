import tensorflow as tf
from ModelCreation import *

model = ResNetModel(HiddenLayers=32, LearningRate=.001, Optimizer="RMSprop", NumFilters=96, Activation="relu", KernelSize=3, Momentum=.9, Epochs=1, BatchSize=512, JobNum=1)

model.load_weights("training_1/cp.ckpt")
X, Y = pickle.load(open("data/400k_training_data.pickle", "rb"))

print(model.summary())

print("Testing:")
model.evaluate(X[:1000], Y[:1000])

print(model.layers[0].get_weights())