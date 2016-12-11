# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import sys

import contexttimer
import numpy as np
import scipy.linalg
import scipy.sparse.linalg

from runlmc.linalg.toeplitz import Toeplitz
from runlmc.linalg.kronecker import Kronecker
import runlmc.util.testing_utils as utils

def stress_kronecker_eig(top, d, exact):
    n = len(top)
    b = np.random.rand(n * d)
    dense = utils.rand_psd(d)
    A = Kronecker(dense, Toeplitz(top))
    M = A.as_numpy()

    assert np.linalg.matrix_rank(dense) == d
    assert np.linalg.matrix_rank(M) == n * d

    exact = bool(exact)
    cutoff = 1e-4
    cond = np.linalg.cond(M)
    dom = np.count_nonzero(top[top > top.max() * cutoff])
    print('    dominant entries {} size {}x{} cutoff {:g} cond {} exact {}'
          .format(dom, n, d, cutoff, cond, exact))

    with contexttimer.Timer() as solve_time:
        eigs = np.linalg.eigvalsh(M)
        eigs[::-1].sort()
        eigs = eigs[eigs > cutoff]
    print('    {} sec {:8.4f}'.format(
        'dense lapack'.rjust(20),
        solve_time.elapsed))

    with contexttimer.Timer() as solve_time:
        my_eigs = A.eig(cutoff, exact)
    print('    {} sec {:8.4f}'.format(
        'kron decomp'.rjust(20),
        solve_time.elapsed))

    if len(my_eigs) != len(eigs):
        print('    INCOMPATIBLE LENGTHS! np {} != mine {}'.format(
            len(eigs), len(my_eigs)))
    minlen = min(len(my_eigs), len(eigs))
    print('    avg eigdiff {} (on shared eigenvalues)'.format(
        np.abs(eigs[:minlen] - my_eigs[:minlen]).mean()))

if __name__ == "__main__":
    if len(sys.argv) not in [4, 5]:
        print('Usage: python eig.py n d exact [seed]')
        print()
        print('n > 8 is the size of the Toeplitz submatrix')
        print('d > 0 is the size of the dense submatrix')
        print('exact is a binary integer')
        print('default seed is 1234')
        print('this eigendecomposes the kronecker product system size n * d')
        print('choose d == 1 and n large to test Toeplitz eig only')
        sys.exit(1)

    n = int(sys.argv[1])
    d = int(sys.argv[2])
    exact = int(sys.argv[3])
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 1234

    assert n > 8
    assert d > 0
    np.random.seed(seed)

    print('* = no convergence')

    print('random (well-cond) ')
    stress_kronecker_eig(utils.random_toep(n), d, exact)

    print('linear decrease (poor-cond)')
    stress_kronecker_eig(utils.poor_cond_toep(n), d, exact)

    print('exponentially decreasing (realistic)')
    stress_kronecker_eig(utils.exp_decr_toep(n), d, exact)
