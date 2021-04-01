import pickle

j = {
    "HiddenLayers": 100,
    "LearningRate": 0.001,
    "Optimizer": "Adam",
    "NumFilters": 64,
    "Activation": "relu",
    "KernelSize": 3,
    "Momentum": .9,
    "Epochs": 10,
    "BatchSize": 64,
    "JobNum": "S",
}

pickle.dump([j], open( "data/jS.pickle", "wb" ))