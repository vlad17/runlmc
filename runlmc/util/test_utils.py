# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import os
import random
import time
import sys

import numpy as np
import scipy.sparse.linalg

from runlmc.linalg.kronecker import Kronecker
from runlmc.linalg.sum_matrix import SumMatrix
from runlmc.linalg.toeplitz import Toeplitz
from runlmc.linalg.numpy_matrix import NumpyMatrix

# I wrote a similar class in databricks/spark-sklearn, but the task is
# small and common enough that the code is basically the same.
class RandomTest:
    """
    This test case mixin sets the random seed to be based on the time
    that the test is run.

    If there is a `SEED` variable in the enviornment, then this is used as the
    seed.

    Sets both random and numpy.random.
    Prints the seed to stdout before running each test case.
    """

    def setUp(self):
        seed = os.getenv("SEED")
        seed = np.uint32(seed if seed else time.time())

        print('Random test using SEED={}'.format(seed))

        random.seed(seed)
        np.random.seed(seed)

def smallest_eig(top):
    A = Toeplitz(top).as_linear_operator()
    try:
        sm = scipy.sparse.linalg.eigsh(
            A, k=1, which='SA', return_eigenvectors=False)[0]
    except scipy.sparse.linalg.eigen.arpack.ArpackNoConvergence:
        sm = np.linalg.eigvalsh(scipy.linalg.toeplitz(top)).min()
    return sm

def poor_cond_toep(n):
    top = np.arange(n)[::-1] * (np.random.rand() + 1e-3)

    while smallest_eig(top) < 0:
        top[0] *= 2

    return top

def random_toep(n):
    top = np.abs(np.random.rand(n))
    top[::-1].sort()
    while smallest_eig(top) < 0:
        top[0] += 1
    return top

def exp_decr_toep(n):
    return np.exp(-(1 + np.random.rand()) * np.arange(n))

def run_main(f, help_str):
    if len(sys.argv) not in [5, 6]:
        print('Usage: python logdet.py n d q eps [seed]')
        print()
        print('n > 8 is the size of the Toeplitz submatrix')
        print('d > 0 is the size of the dense submatrix')
        print('q > 0 is the number of dense-Toeplitz Kronecker products')
        print('      to sum together for the system')
        print('eps >= 0 is the constant diagonal perturbation (a float)')
        print('         added in (higher eps -> better conditioning).')
        print('default seed is 1234')
        print()
        print(help_str)
        print()
        print('Choose q = d = 1 and n large to test Toeplitz, mainly')
        print('Choose q = 1 and n ~ d^2 > 1 to test Kronecker, mainly')
        sys.exit(1)

    n = int(sys.argv[1])
    d = int(sys.argv[2])
    q = int(sys.argv[3])
    eps = float(sys.argv[4])
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 1234

    assert n > 8
    assert d > 0
    assert q > 0
    assert eps >= 0
    np.random.seed(seed)

    print('size q {} n {} d {} eps {:g}'.format(q, n, d, eps))

    cases = [
        ('random (well-cond) ', random_toep),
        ('linear decrease (poor-cond)', poor_cond_toep),
        ('exponentially decreasing (realistic)', exp_decr_toep)]

    noise = np.ones(n * d) * eps

    for name, generator in cases:
        print(name)
        dense_mats = [rand_psd(d) for _ in range(q)]
        toep_tops = [generator(n) for _ in range(q)]
        my_mat = SumMatrix([Kronecker(NumpyMatrix(dense), Toeplitz(top))
                            for dense, top in zip(dense_mats, toep_tops)],
                           noise)
        np_mat = sum(np.kron(dense, scipy.linalg.toeplitz(top))
                     for dense, top in zip(dense_mats, toep_tops))
        np_mat += np.diag(noise)
        f(my_mat, np_mat, n, d, q, eps)

def rand_psd(n):
    A = np.random.rand(n, n)
    A = (A + A.T).astype(np.float64) / 2
    A += np.diag(np.fabs(A).sum(axis=1) + 1)
    return A
