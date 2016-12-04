# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import sys

import contexttimer
import numpy as np
import scipy.linalg
import scipy.sparse.linalg

from runlmc.linalg.toeplitz import Toeplitz
from runlmc.linalg.kronecker import Kronecker
import runlmc.util.test_utils as utils

def stress_kronecker_eig(top, d):
    n = len(top)
    b = np.random.rand(n * d)
    toep = scipy.linalg.toeplitz(top)
    dense = utils.rand_psd(d)

    assert np.linalg.matrix_rank(dense) == d
    assert np.linalg.matrix_rank(toep) == n

    A = Kronecker(dense, Toeplitz(top))
    M = np.kron(dense, toep)
    assert np.linalg.matrix_rank(M) == n * d

    cutoff = 1e-3
    cond = np.linalg.cond(M)
    print('    size {}x{} cutoff {:g} cond {}'
          .format(n, d, cutoff, cond))

    with contexttimer.Timer() as solve_time:
        eigs = np.linalg.eigvalsh(M)
        eigs[::-1].sort()
        eigs = eigs[eigs > cutoff]
    print('    {} sec {:8.4f}'.format(
        'dense lapack'.rjust(20),
        solve_time.elapsed))

    with contexttimer.Timer() as solve_time:
        my_eigs = A.eig(cutoff)
    print('    {} sec {:8.4f}'.format(
        'kron decomp'.rjust(20),
        solve_time.elapsed))

    if len(my_eigs) != len(eigs):
        print('    INCOMPATIBLE LENGTHS! np {} != mine {}'.format(
            len(eigs), len(my_eigs)))
    else:
        print('    avg eigdiff {}'.format(
            np.abs(eigs - my_eigs).mean()))

if __name__ == "__main__":
    if len(sys.argv) not in [3, 4]:
        print('Usage: python eig.py n d [seed]')
        print()
        print('n > 8 is the size of the Toeplitz submatrix')
        print('d > 0 is the size of the dense submatrix')
        print('default seed is 1234')
        print('this eigendecomposes the kronecker product system size n * d')
        print('choose d == 1 and n large to test Toeplitz eig only')
        sys.exit(1)

    n = int(sys.argv[1])
    d = int(sys.argv[2])
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1234

    assert n > 8
    assert d > 0
    np.random.seed(seed)

    print('* = no convergence')

    print('random (well-cond) ')
    stress_kronecker_eig(utils.random_toep(n), d)

    print('linear decrease (poor-cond)')
    stress_kronecker_eig(utils.poor_cond_toep(n), d)

    print('exponentially decreasing (realistic)')
    stress_kronecker_eig(utils.exp_decr_toep(n), d)
