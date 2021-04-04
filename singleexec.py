import sys
import resnet
import pandas as pd
import tensorflow as tf
import pickle


job_n = sys.argv[1]
gpu_n = sys.argv[2]

jobs = pickle.load( open( f"data/j{job_n}.pickle", "rb" ) )

data = pd.DataFrame(
    columns=["HiddenLayers", "LearningRate", "Optimizer",
            "NumFilters", "Activation", "KernelSize",
            "Momentum", "Epochs", "Loss",
            "Acc", "Val_Loss", "Val_Acc",]
)

filename = f"data/{job_n}_results.csv"
data.to_csv(filename, header=True)

print(f"Starting job #{job_n} on gpu #{gpu_n}")

try:
    for j in range(len(jobs)):
        print(f"Starting job #{j} of {len(jobs) - 1}\n")
        resnet.ResNet(**jobs[j]).to_csv(filename, mode='a', header=False)
        print(f"Done with job # {j} or {len(jobs) - 1}\n")
except:
    print("interrupted! trying to save data")
    sys.exit(0)
