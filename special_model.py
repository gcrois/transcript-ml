import pickle

j = {
    "HiddenLayers": 100,
    "LearningRate": 0.05,
    "Optimizer": "RMSprop",
    "NumFilters": 96,
    "Activation": "relu",
    "KernelSize": 5,
    "Momentum": .9,
    "Epochs": 10,
    "BatchSize": 64,
}

pickle.dump([j], open( "data/jS.pickle", "wb" ))