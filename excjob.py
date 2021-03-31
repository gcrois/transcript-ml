import sys
import resnet
import pandas as pd
import pickle


jobs = pickle.load( open( f"data/j{sys.argv[1]}.pickle", "rb" ) )

data = pd.DataFrame(
    columns=["HiddenLayers", "LearningRate", "Optimizer",
            "NumFilters", "Activation", "KernelSize",
            "Momenetum", "Epochs", "Time",
            "Loss", "Acc", "Val_Loss",
            "Val_Acc",]
)

for j in jobs:
    print("Starting job\n", j)
    data = data.append(resnet.ResNet(**j))
    print("Done with job\n")

data.to_csv(f"data/{jobs[0]['j_number']}_results.csv")