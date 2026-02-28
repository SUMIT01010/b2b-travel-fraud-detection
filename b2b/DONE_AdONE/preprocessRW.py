from scipy.sparse import diags
import scipy as sp
import numpy as np


def computeRep(G, K, c):
    """
    Optimized implementation of the Personalized PageRank-like representation.
    Uses dense NumPy operations for much better performance on the 8000x8000 dense G.
    """
    A = G
    n = G.shape[0]
    
    # Calculate degree and inverse degree
    degree = np.sum(G, axis=1)
    # Avoid division by zero
    degree[degree == 0] = 0.1
    inv_degree = 1.0 / degree
    
    # Transition matrix P = D^-1 * A
    # Broadcasting multiplication is equivalent to InvDegree @ A
    intermedMat = inv_degree[:, np.newaxis] * A
    
    # Initialize P as identity matrix (P_0)
    P_0 = np.eye(n)
    P = P_0
    
    # Iterative calculation of Personalized PageRank: P = c * P * intermedMat + (1-c) * I
    # We can use dense matrix multiplication (@) which is highly optimized
    print(f"Starting iterative preprocessing (K={K}, c={c})...")
    for k in range(K):
        # Using @ instead of sparse multiplication is much faster for dense 8000x8000 matrices
        P = c * (P @ intermedMat) + (1 - c) * P_0
        
    return P