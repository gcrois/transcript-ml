import sys
import resnet
import pandas as pd
import tensorflow as tf
import pickle


jobs = pickle.load( open( f"data/j{sys.argv[1]}.pickle", "rb" ) )

data = pd.DataFrame(
    columns=["HiddenLayers", "LearningRate", "Optimizer",
            "NumFilters", "Activation", "KernelSize",
            "Momentum", "Epochs", "Loss",
            "Acc", "Val_Loss", "Val_Acc",]
)

with tf.device(f'/GPU:{sys.argv[2]}'):
    for j in range(len(jobs)):
        print("Starting job #{j}\n", j)
        data = data.append(resnet.ResNet(**jobs[j]))
        print("Done with job # {j}\n")

data.to_csv(f"data/{jobs[0]['JobNum']}_results.csv")