# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import numpy as np
from scipy.fftpack import fft, ifft
import scipy.linalg as la
import scipy.sparse.linalg as sla

from ..approx.ski import SKI
from ..linalg.symmetric_matrix import SymmetricMatrix
from ..linalg.toeplitz import Toeplitz
from ..linalg.kronecker import Kronecker
from ..linalg.sum_matrix import SumMatrix
from ..util.docs import inherit_doc

class GridKernel(SymmetricMatrix):
    def __init__(self, params, grid_dists, interpolant, interpolantT):
        super().__init__(params.n)
        self.params = params
        self.dists = grid_dists
        self.interpolant = interpolant
        self.interpolantT = interpolantT

    def grid_only(self):
        raise NotImplementedError

def gen_grid_kernel(params, grid_dists, interpolant, interpolantT):
    sum_grid = SumGridKernel(params, grid_dists, interpolant, interpolantT)
    return sum_grid

def mkpre(params, grid, interpolant, interpolantT):
    bt = _toep_blocks(params.kernels, params.coreg_mats, grid, params.D)
    grid_lo = _chan_bt_pre(bt)
    return SKI(grid_lo, interpolant, interpolantT)

def _toep_blocks(kernels, coreg_mats, grid_dists, D):
    tops = np.array([k.from_dist(grid_dists) for k in kernels])
    z = np.zeros_like(tops[0])
    z[0] += 1
    tops = np.concatenate((tops, [z]))
    Bs = np.array(coreg_mats + [np.identity(D)])
    return np.tensordot(Bs, tops, axes=(0, 0))

def _np_map2(f, x):
    return [[f(x[i, j]) for j in range(x.shape[1])]for i in range(x.shape[0])]

def _chan(t): # could be parallel...
    n = len(t)
    inc = np.arange(n)
    l = t * (n - inc)
    l = l.astype(float)
    r = t[::-1][:-1] * inc[1:]
    l[1:] += r
    return l / n

def _inv_precond(lus, perm, inv_perm, x):
    D = lus[0][0].shape[0]
    x = x.astype('complex').reshape(D, -1)
    x = fft(x, overwrite_x=True).reshape(-1)
    x = x[perm].reshape(-1, D)
    x = np.hstack([la.lu_solve(factors, b) for factors, b in zip(lus, x)])
    x = x[inv_perm].reshape(D, -1)
    return ifft(x, overwrite_x=True).reshape(-1)

def _chan_bt_pre(toep_blocks):
    sz = toep_blocks.shape[1] * toep_blocks.shape[2]
    circ_pre = np.array(_np_map2(_chan, toep_blocks))
    circ_eigs = fft(circ_pre, overwrite_x=True) # applies to last axis
    rc = np.rollaxis(circ_eigs, -1, 0)
    lus = [la.lu_factor(dd, overwrite_a=True) for dd in rc]
    # the rolled axis perm
    perm = np.add.outer(
        np.arange(circ_eigs.shape[2]),
        np.arange(circ_eigs.shape[0]) * circ_eigs.shape[2]).ravel()
    inv_perm = np.zeros(len(perm), dtype=int)
    inv_perm[perm] = np.arange(len(perm))
    lx = lambda x: _inv_precond(lus, perm, inv_perm, x).real
    return sla.LinearOperator((sz, sz), matvec=lx, rmatvec=lx)

# TODO(SLFM-representation)
# TODO(block-toeplitz)

@inherit_doc
class SumGridKernel(GridKernel):
    def __init__(self, params, grid_dists, interpolant, interpolantT):
        super().__init__(params, grid_dists, interpolant, interpolantT)
        kerns_on_grid = [Toeplitz(k.from_dist(grid_dists))
                         for k in params.kernels]
        products = [Kronecker(A, K) for A, K in
                    zip(params.coreg_mats, kerns_on_grid)]
        ksum = SumMatrix(products)
        self.ski = SKI(ksum, interpolant, interpolantT)
        self.noise = np.repeat(params.noise, params.lens)

    def matvec(self, x):
        return self.ski.matvec(x) + self.noise * x

    def as_numpy(self):
        return self.ski.as_numpy() + np.diag(self.noise)

    def upper_eig_bound(self):
        return self.ski.upper_eig_bound() + self.noise.max()

    def grid_only(self):
        return self.ski.K

@inherit_doc
class SumGridKernel(GridKernel):
    def __init__(self, params, grid_dists, interpolant, interpolantT):
        super().__init__(params, grid_dists, interpolant, interpolantT)
        kerns_on_grid = [Toeplitz(k.from_dist(grid_dists))
                         for k in params.kernels]
        products = [Kronecker(A, K) for A, K in
                    zip(params.coreg_mats, kerns_on_grid)]
        ksum = SumMatrix(products)
        self.ski = SKI(ksum, interpolant, interpolantT)
        self.noise = np.repeat(params.noise, params.lens)

    def matvec(self, x):
        return self.ski.matvec(x) + self.noise * x

    def as_numpy(self):
        return self.ski.as_numpy() + np.diag(self.noise)

    def upper_eig_bound(self):
        return self.ski.upper_eig_bound() + self.noise.max()

    def grid_only(self):
        return self.ski.K
