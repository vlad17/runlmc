# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import contexttimer
import numpy as np
import scipy.linalg
import scipy.sparse.linalg

from runlmc.linalg.kronecker import Kronecker
from runlmc.linalg.sum_matrix import SumMatrix
from runlmc.linalg.toeplitz import Toeplitz
from runlmc.linalg.numpy_matrix import NumpyMatrix
import runlmc.util.testing_utils as utils

def stress_logdet(my_mat):
    np_mat = my_mat.as_numpy()
    cond = np.linalg.cond(np_mat)
    print('    cond {}'.format(cond))

    with contexttimer.Timer() as solve_time:
        sign, logdet = np.linalg.slogdet(np_mat)
    assert sign > 0, sign
    print('    {} sec {:8.4f} logdet {:10.4g}'.format(
        'dense lapack'.rjust(20),
        solve_time.elapsed,
        logdet))

    with contexttimer.Timer() as solve_time:
        my_logdet = my_mat.logdet()
    print('    {} sec {:8.4f} logdet {:10.4g}'.format(
        'logdetsum'.rjust(20),
        solve_time.elapsed,
        my_logdet))

    rel_err = abs(logdet - my_logdet) / abs(logdet)
    print('    relative error {}'.format(rel_err))

HELP_STR = ('This benchmarks determinant finding of a system of a sum of\n'
            'dense-Toeplitz Kronecker products, which has size n * d\n'
            'but only n * d^2 * q parameters, as opposed to\n'
            'the dense system of (n * d)^2 parameters')

if __name__ == "__main__":
    utils.run_main(stress_logdet, HELP_STR)
