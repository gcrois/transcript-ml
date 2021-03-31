from itertools import product
from definitions import *
import pickle
import random

prod = [dict(zip(PARAMETERS, v)) for v in product(*PARAMETERS.values())]

jobs = [prod[i::N_JOBS] for i in range(N_JOBS)]

random.shuffle(jobs)

for j in range(N_JOBS):
    for i in jobs[j]:
        i["j_number"] = j

print(jobs[0][0], jobs[1][0], jobs[2][0], jobs[3][0])

for j in range(N_JOBS):
    pickle.dump(jobs[j], open( f"data/j{j}.pickle", "wb" ))