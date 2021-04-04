"""
    This file is a copy of the excjob.py except it is
    for the use of the alternative resnet model
"""

import sys
import resnetAlt
import pandas as pd
import tensorflow as tf
import pickle

jobs = pickle.load(open("data/training_data.pickle", "rb"))

data = pd.DataFrame(
    columns=["Epochs", "Loss", "Acc", "Val_Loss", "Val_Acc"]
)

filename = "data_results_Alt.csv"
data.to_csv(filename, header=True)

resnetAlt.ResNetAlt(10).to_csv(filename, mode='a', header=False)

print(f"Starting job")

#with tf.device(f'/GPU:{gpu_n}'):
