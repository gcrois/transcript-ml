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

filename = f"data/{jobs[0]['JobNum']}_results.csv"
data.to_csv('my_csv.csv', mode='a', header=True)

with tf.device(f'/GPU:{sys.argv[2]}'):
    try:
        for j in range(len(jobs)):
            print(f"Starting job #{j}\n")
            resnet.ResNet(**jobs[j]).to_csv('my_csv.csv', mode='a', header=False)
            print(f"Done with job # {j}\n")
    except:
        print("interrupted! trying to save data")
        sys.exit(0)