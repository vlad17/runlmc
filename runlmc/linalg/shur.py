# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import numpy as np

# Cholesky factorization of semidefinite Toeplitz matrices
# Michael Stewart 1997
# returns upper cholesky of a toeplitz matrix
def shur(top):
    n = len(top)
    G = np.zeros((2, n))
    G[0] = top / top[0]
    G[1] = G[0]
    G[1, 0] = 0

    C = np.zeros((n, n))
    C[0] = G[0]

    G[0, 1:] = np.copy(G[0, :-1])
    G[0, 0] = 0
    for i in range(1, n):
        rho = -G[1, i] / G[0, i]
        if abs(rho) >= 1:
            print('rank def', 'rho', rho, 'i', i, 'n', n)
            break
        scale = np.sqrt(1 - rho) * np.sqrt(1 + rho) # taylor for stability?
        G[:, i:] = np.array([[1, rho], [rho, 1]]).dot(G[:, i:])
        G[:, i:] /= scale
        C[i, i:] = G[0, i:]
        G[0, (i+1):] = np.copy(G[0, i:-1])
        G[0, i] = 0
    return C * np.sqrt(top[0])
