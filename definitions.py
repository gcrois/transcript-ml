classification = {
    "Alpha" : 0,
    "Beta" : 1,
    "Chi" : 2,
    "Delta" : 3,
    "Epsilon" : 4,
    "Eta" : 5,
    "Gamma" : 6,
    "Iota" : 7,
    "Kappa" : 8,
    "Lambda" : 9,
    "Mu" : 10,
    "Nu" : 11,
    "Omega" : 12,
    "Omicron" : 13,
    "Phi" : 14,
    "Pi" : 15,
    "Psi" : 16,
    "Rho" : 17,
    "Sigma" : 18,
    "Tau" : 19,
    "Theta" : 20,
    "Upsilon" : 21,
    "Xi" :22,
    "Zeta" : 23
}

PARAMETERS = {
    "layers": set([2, 4, 8, 16, 32, 50]),
    "rates": set([0.001, 0.01, 0.03, 0.05]),
    "optimizer": set(["ADAM", "rmsprop"]),
    "filters": set([16, 32, 64, 96]),
    "activation": set(["RELU"]),
    "kernel_size": set([3]),
    "momentum": set([.9]),
    "epochs": set([75]),
}

ROWS = 70
COLS = 70
CHANNELS = 3
CLASSES = 24
LENGTH = 1000
BLOCKS = 50
EPOCHS = 75
N_JOBS = 4