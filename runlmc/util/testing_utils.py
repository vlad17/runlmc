# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

"""
The following methods are useful for generating various matrices for testing.

"PSDT" stands for positive semi-definite Toeplitz (and, implicitly, symmetric).
"""

import os
import random
import time
import sys
import unittest

import numpy as np
from paramz.optimization import Optimizer

from ..linalg.kronecker import Kronecker
from ..linalg.sum_matrix import SumMatrix
from ..linalg.toeplitz import Toeplitz
from ..linalg.numpy_matrix import NumpyMatrix
from .numpy_convenience import smallest_eig

class RandomTest(unittest.TestCase):
    """
    This test case sets the random seed to be based on the time
    that the test is run.

    If there is a `SEED` variable in the enviornment, then this is used as the
    seed.

    Sets both random and numpy.random.
    Prints the seed to stdout before running each test case.
    """

    def setUp(self):
        super().setUp()

        seed = os.getenv("SEED")
        if seed is None:
            seed = int(time.time() * 37 + os.getpid())
        self.seed = np.array([seed]).astype(np.uint32)[0]

        print('Random test using SEED={}'.format(self.seed))

        random.seed(self.seed)
        np.random.seed(self.seed)

def poor_cond_toep(n):
    """
    :param n: size of output
    :returns: the top row of a randomly scaled PSDT matrix whose
              :math:`L^2` condition number scales exponentially with `n`
    """
    top = np.arange(n)[::-1] * (np.random.rand() + 1e-3)

    while smallest_eig(top) < 0:
        top[0] *= 2

    return top

def random_toep(n):
    """
    :returns: top row of a random PSDT matrix of size `n`.
    """
    top = np.abs(np.random.rand(n))
    top[::-1].sort()
    while smallest_eig(top) < 0:
        top[0] += 1
    return top

def exp_decr_toep(n):
    """
    :returns: top row of a PSDT matrix of size `n` with terms
              exponentially decreasing in distance from the main diagonal;
              the rate of which is randomly generated but at least
              :math:`e`.
    """
    return np.exp(-(1 + np.random.rand()) * np.arange(n))

def run_main(f, help_str):
    """
    A helper function which is re-used in setting up benchmarks for kernels
    that are shaped in a specific manner; namely, sums of Kronecker products
    of small dense and large Toeplitz matrices.

    This function reads in command-line arguments and executes a simple
    benchmarking script.

    :param f: benchmarking function to pass generated inputs with
              user-specified parameters to.
    :param help_str: a help-string to be printed when the user does not
                     call a correct invocation of the program.
    """
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
        f(my_mat)

def rand_psd(n):
    """
    :returns: a random `n` by `n` symmetric PSD matrix with positive entries
    """
    A = np.random.rand(n, n)
    A = (A + A.T) / 2
    D = np.diag(np.fabs(A).sum(axis=1) + 1)
    return A + D


def check_np_lists(a, b, atol=0):
    """
    Verifies that two lists of numpy arrays are all close.
    :param a:
    :param b:
    """
    assert len(a) == len(b), 'a {} b {}'.format(len(a), len(b))
    for i, (sub_a, sub_b) in enumerate(zip(a, b)):
        np.testing.assert_allclose(
            sub_a, sub_b, err_msg='output {}'.format(i), atol=atol)

class SingleGradOptimizer(Optimizer):
    def __init__(self):
        super().__init__()
        self.gradient_observed = None

    def opt(self, x_init, f_fp=None, f=None, fp=None):
        # 1 iteration only
        # Note we save the negative of the gradient, since
        # the Model class will implicitly flip the objective's sign
        # to make the likelihood maximization into a minimization problem.
        self.gradient_observed = -fp(x_init)
        self.x_opt = x_init + self.gradient_observed
