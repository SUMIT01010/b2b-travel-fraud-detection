import numpy as np
import pandas as pd
import scipy.sparse as sp
import os

edges = np.loadtxt("data/graph/edges.txt", dtype=int)
X = np.load("data/graph/features.npy")

n = X.shape[0]

adj = sp.coo_matrix(
    (np.ones(edges.shape[0]), (edges[:,0], edges[:,1])),
    shape=(n,n)
).todense()

os.makedirs("DONE_AdONE/Data/b2b", exist_ok=True)

pd.DataFrame(adj).to_csv(
    "DONE_AdONE/Data/b2b/A_Final_permuted.csv",
    index=False,
    header=False
)

pd.DataFrame(X).to_csv(
    "DONE_AdONE/Data/b2b/C_Final_permuted.csv",
    index=False,
    header=False
)

print("B2B graph exported for DONE")
