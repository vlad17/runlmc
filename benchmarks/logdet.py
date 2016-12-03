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

def random_toep(n):
    top = np.abs(np.random.rand(n))
    top[::-1].sort()
    top[0] += 1
    return top

def poor_cond_toep(n):
    top = np.arange(n)[::-1] * (np.random.rand() + 1e-3)

    def smallest_eig(T):
        A = T.as_linear_operator()
        try:
            sm = scipy.sparse.linalg.eigsh(
                A, k=1, which='SA', return_eigenvectors=False)[0]
        except scipy.sparse.linalg.eigen.arpack.ArpackNoConvergence:
            sm = np.linalg.eigvalsh(scipy.linalg.toeplitz(T.top)).min()
        return sm

    changed = False
    while True:
        T = Toeplitz(top)
        sm = smallest_eig(T)
        if sm >= 0:
            if changed:
                print('        smallest eig {:8.4f} >= 0'
                  .format(sm))
            break
        changed = True
        print('        smallest eig {:8.4f} < 0, doubling main diagonal!'
              .format(sm))
        top[0] *= 2

    return top

def exp_decr_toep(n):
    return np.exp(-(1 + np.random.rand()) * np.arange(n))

def rand_psd(n):
    A = np.random.rand(n, n)
    A = (A + A.T).astype(np.float64) / 2
    A += np.diag(np.fabs(A).sum(axis=1) + 1)
    return A

def stress_sum_solve(top_gen, n, d, q, noise):
    b = np.random.rand(n * d)

    dense_mats = [rand_psd(d) for _ in range(q)]
    toep_tops = [top_gen(n) for _ in range(q)]
    A = SumMatrix([Kronecker(NumpyMatrix(dense), Toeplitz(top))
                   for dense, top in zip(dense_mats, toep_tops)],
                  noise)
    M = sum(np.kron(dense, scipy.linalg.toeplitz(top))
            for dense, top in zip(dense_mats, toep_tops))
    M += np.diag(noise)
    assert np.linalg.matrix_rank(M) == n * d

    # err is same formula as SumMatix uses
    err = max(noise.min(), 1e-10 * q * n * d)
    cond = np.linalg.cond(M)
    print('    size qxnxd {}x{}x{} err {:g} cond {}'
          .format(q, n, d, err, cond))

    with contexttimer.Timer() as solve_time:
        sign, logdet = np.linalg.slogdet(M)
    assert sign > 0, sign
    print('    {} sec {:8.4f} logdet {:10.4g}'.format(
        'dense lapack'.rjust(20),
        solve_time.elapsed,
        logdet))

    with contexttimer.Timer() as solve_time:
        my_logdet = A.logdet()
    print('    {} sec {:8.4f} logdet {:10.4g}'.format(
        'logdetsum'.rjust(20),
        solve_time.elapsed,
        my_logdet))


if __name__ == "__main__":
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
        print('This benchmarks determinant finding of a system of a sum of\n'
              'dense-Toeplitz Kronecker products, which has size n * d\n'
              'but only n * d^2 * q parameters, as opposed to\n'
              'the dense system of (n * d)^2 parameters')
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

    noise = np.ones(n * d) * eps

    print('* = no convergence')

    print('random (well-cond) ')
    stress_sum_solve(random_toep, n, d, q, noise)

    # Poorly-conditioned
    print('linear decrease (poor-cond)')
    stress_sum_solve(poor_cond_toep, n, d, q, noise)

    print('exponentially decreasing (realistic)')
    stress_sum_solve(exp_decr_toep, n, d, q, noise)
