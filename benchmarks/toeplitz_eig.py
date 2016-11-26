# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import sys
import warnings

import contexttimer
import numpy as np
import scipy.linalg
import scipy.sparse.linalg

from runlmc.linalg.toeplitz import SymmToeplitz

def extended_logdet(vals, sigma, n):
    return np.log(vals).sum() + np.log(sigma) * (n - len(vals))

def stress_toeplitz_eig(top):
    n = len(top)
    toep = scipy.linalg.toeplitz(top)
    assert np.linalg.matrix_rank(toep) == n
    A = SymmToeplitz(top)

    tol = 1e-15

    print('Toeplitz Eigenvalues: size {} tol {:g} cond {}'
          .format(n, tol, np.linalg.cond(toep)))

    with contexttimer.Timer() as eigsh_time:
        vals = A.eig(tol)
    print('    Sparse eigsh  {:4.4f} s'.format(eigsh_time.elapsed))

    with contexttimer.Timer() as exact_time:
        np_vals = np.linalg.eigvalsh(toep)
    print('    Dense eigs    {:4.4f} s'.format(exact_time.elapsed))
    np_vals[::-1].sort()
    np_vals = np_vals[np_vals > tol]

    assert all(np.diff(vals) <= 0)
    assert all(np.diff(np_vals) <= 0)
    assert len(vals) == sum(vals > tol)

    print('    Eigs > tol found: {} of {}'.format(
        len(vals), sum(np_vals > tol)))

    noise = 1e-3
    ld_vals = extended_logdet(vals, noise, n)
    ld_np_vals = extended_logdet(np_vals, noise, n)
    print('    Soft log det error (noise {:g}) {}'.format(
        noise, abs(ld_vals - ld_np_vals) / abs(ld_np_vals)))
    # TODO: need to figure out approx multiplicity.

    np.testing.assert_allclose(vals, np_vals[:len(vals)], atol=1e-6, rtol=0)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python toeplitz_eig.py n')
        print()
        print('n > 3 should be size of linear systems to solve')
        sys.exit(1)

    n = int(sys.argv[1])
    assert n > 3

    # Well-conditioned

    top = np.random.rand(n)
    b = np.random.rand(n)
    top[::-1].sort()
    top[0] * 2
    #top[0] += np.abs(top).sum() + 1 # make invertible
    stress_toeplitz_eig(top)

    # Poorly-conditioned

    up = np.arange(n // 3) + 1
    down = np.copy(up[::-1])
    up = up / 2
    top = np.zeros(n)
    updown = np.add.accumulate(np.hstack([up, down]))[::-1]
    top[:len(updown)] = updown
    stress_toeplitz_eig(top)
