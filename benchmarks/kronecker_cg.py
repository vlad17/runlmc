# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import sys

import contexttimer
import numpy as np
import scipy.linalg
import scipy.sparse.linalg

from runlmc.linalg.toeplitz import SymmToeplitz
from runlmc.linalg.kronecker import Kronecker

def stress_kronecker_solve(top, d):
    n = len(top)
    b = np.random.rand(n * d)
    toep = scipy.linalg.toeplitz(top)

    D = np.random.rand(d, d)
    D += D.T
    D += np.diag(np.ones(d) * (np.abs(D).sum() + 1))

    assert np.linalg.matrix_rank(D) == d
    assert np.linalg.matrix_rank(toep) == n

    T = SymmToeplitz(top)
    A = scipy.sparse.linalg.aslinearoperator(Kronecker(D, T))
    M = np.kron(D, toep)
    assert np.linalg.matrix_rank(M) == n * d

    tol = 1e-6

    print('    size {}x{} tol {:g} cond {}'
          .format(n, d, tol, np.linalg.cond(M)))
    ctr = 0
    def inc(_):
        nonlocal ctr
        ctr += 1
    with contexttimer.Timer() as cg_sparse_time:
        cg_sparse, success = scipy.sparse.linalg.cg(
            A, b, tol=tol, callback=inc, maxiter=(n*d))
    print('    Sparse Kron CG {:4.4f} s ({} iterations)'
          .format(cg_sparse_time.elapsed, ctr))

    with contexttimer.Timer() as cg_dense_time:
        cg_dense, success = scipy.sparse.linalg.cg(M, b, tol=tol)
        assert success == 0
        print('    Dense CG       {:4.4f} s'.format(cg_dense_time.elapsed))

    with contexttimer.Timer() as linsolve_time:
        linsolve = np.linalg.solve(M, b)
    print('    Linear solve   {:4.4f} s'.format(linsolve_time.elapsed))

    np.testing.assert_allclose(cg_sparse, linsolve, atol=(tol * 2))
    np.testing.assert_allclose(cg_sparse, cg_dense, atol=(tol * 2))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('Usage: python kronecker_cg.py n d')
        print()
        print('n > 8 is the size of the Toeplitz submatrix')
        print('d > 0 is the size of the dense submatrix')
        print('this solves the kronecker product system size n * d')
        sys.exit(1)

    n = int(sys.argv[1])
    d = int(sys.argv[2])

    np.random.seed(1234)

    # Well-conditioned
    assert n > 8
    assert d > 0

    top = np.random.rand(n)
    b = np.random.rand(n)
    top[::-1].sort()
    top[0] *= 2
    print('random (well-cond) ')
    stress_kronecker_solve(top, d)

    # Poorly-conditioned
    print('skipping poorly-conditioned case:')
    print('    it is unrealistic and breaks CG')

    print('exponentially decreasing (realistic)')
    stress_kronecker_solve(np.exp(-np.arange(n)), d)
