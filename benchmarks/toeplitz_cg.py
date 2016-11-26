# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import sys

import contexttimer
import numpy as np
import scipy.linalg
import scipy.sparse.linalg

from runlmc.linalg.toeplitz import SymmToeplitz

def stress_toeplitz_solve(n):
    top = np.random.rand(n)
    b = np.random.rand(n)
    top[0] += np.abs(top).sum() + 1 # make invertible
    toep = scipy.linalg.toeplitz(top)
    assert np.linalg.matrix_rank(toep) == n


    A = scipy.sparse.linalg.aslinearoperator(SymmToeplitz(top))
    tol = 1e-15

    print('Toeplitz Solver: size {} tol {:g}'.format(n, tol))

    ctr = 0
    def inc(_):
        nonlocal ctr
        ctr += 1
    with contexttimer.Timer() as cg_sparse_time:
        cg_sparse, success = scipy.sparse.linalg.cg(
            A, b, tol=tol, callback=inc)
        assert success == 0
    print('    Sparse FFT CG {:4.4f} s ({} iterations)'
          .format(cg_sparse_time.elapsed, ctr))

    with contexttimer.Timer() as cg_dense_time:
        cg_dense, success = scipy.sparse.linalg.cg(toep, b, tol=tol)
        assert success == 0
    print('    Dense CG      {:4.4f} s'.format(cg_dense_time.elapsed))

    with contexttimer.Timer() as linsolve_time:
        linsolve = np.linalg.solve(toep, b)
    print('    Linear solve  {:4.4f} s'.format(linsolve_time.elapsed))

    np.testing.assert_allclose(cg_sparse, linsolve, atol=(tol * 2))
    np.testing.assert_allclose(cg_sparse, cg_dense, atol=(tol * 2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python toeplitz_cg.py n')
        print()
        print('n should be size of linear systems to solve')
        sys.exit(1)

    n = int(sys.argv[1])
    stress_toeplitz_solve(n)
