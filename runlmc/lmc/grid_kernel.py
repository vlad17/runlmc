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
    # bt = _toep_blocks(params.kernels, params.coreg_mats, grid, params.D)
    sum_grid = SumGridKernel(params, grid_dists, interpolant, interpolantT)
    return sum_grid

def _toep_blocks(kernels, coreg_mats, grid_dists, D):
    tops = np.array([k.from_dist(grid_dists) for k in kernels])
    z = np.zeros_like(tops[0])
    z[0] += 1
    tops = np.concatenate((tops, [z]))
    Bs = np.array(coreg_mats + [np.identity(D)])
    return np.tensordot(Bs, tops, axes=(0, 0))

# TODO(cleanup): move to numpy convenience - make efficient / numpy iteration?
def _np_map2(f, x):
    return [[f(x[i, j]) for j in range(x.shape[1])]for i in range(x.shape[0])]

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
