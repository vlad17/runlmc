# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import sys

import contexttimer
import numpy as np
import scipy.linalg
import scipy.sparse.linalg

from runlmc.linalg.kronecker import Kronecker
from runlmc.linalg.sum_matrix import SumMatrix
from runlmc.linalg.toeplitz import Toeplitz
from runlmc.linalg.numpy_matrix import NumpyMatrix
import runlmc.util.testing_utils as utils

def stress_sum_solve(my_mat):
    b = np.random.rand(my_mat.shape[0])
    sz = len(b)
    np_mat = my_mat.as_numpy()
    linop = my_mat.as_linear_operator()

    tol = 1e-6
    cond = np.linalg.cond(np_mat)
    print('    cond {} tol {:g}'.format(cond, tol))

    def time_method(f):
        with contexttimer.Timer() as solve_time:
            solve, name = f()
        print('    {} sec {:8.4f} resid {:8.4e}'.format(
            name.rjust(20),
            solve_time.elapsed,
            np.linalg.norm(linop.matvec(solve) - b)))
    time_method(lambda: (np.linalg.solve(np_mat, b), 'linear solve'))

    def sparse():
        out, succ = scipy.sparse.linalg.cg(my_mat, b, tol=tol, maxiter=sz)
        return out, '{} sparse CG'.format('' if not succ else '*')
    time_method(sparse)

    def minres():
        out, succ = scipy.sparse.linalg.minres(my_mat, b, tol=tol,
                                               maxiter=sz)
        return out, '{} sparse MINRES'.format('' if not succ else '*')
    time_method(minres)

HELP_STR = ('This benchmarks solving a system defined by a sum of\n'
            'dense-Toeplitz Kronecker products, which has size n * d\n'
            'but only n * d^2 * q parameters, as opposed to\n'
            'the dense system of (n * d)^2 parameters')

if __name__ == "__main__":
    print('* = no convergence')
    utils.run_main(stress_sum_solve, HELP_STR)
