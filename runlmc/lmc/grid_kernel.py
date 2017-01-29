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

# TODO(SLFM-representation)

def _symm_2d_list_map(f, arr, D):
    out = np.empty((D, D), dtype='object')
    for i in range(D):
        for j in range(i, D):
            out[i, j] = f(arr[i, j])
            out[j, i] = out[i, j]
    return out

# TODO(cleanup): move to linalg
@inherit_doc
class BlockMatrix(SymmetricMatrix):
    def __init__(self, blocks):
        super().__init__(len(blocks) * blocks[0][0].shape[0])
        self.blocks = blocks
        self.D = len(blocks)

    def matvec(self, x):
        # potential optimization - rm split, use slices
        shards = np.split(x, self.D)
        result = np.split(np.zeros_like(x), self.D)
        for i, row in enumerate(self.blocks):
            for j, shard in enumerate(shards):
                result[i] += row[j].matvec(shard)
        return np.hstack(result)

    def as_numpy(self):
        mats = _symm_2d_list_map(lambda x: x.as_numpy(), self.blocks, self.D)
        return np.bmat(mats).A

    def upper_eig_bound(self):
        bounds = _symm_2d_list_map(lambda x: x.upper_eig_bound(),
                                   self.blocks, self.D)
        bounds = bounds.astype(float)
        return np.linalg.norm(bounds, 1)

@inherit_doc
class BlockToeplitz(GridKernel):
    def __init__(self, params, grid_dists, interpolant, interpolantT):
        super().__init__(params, grid_dists, interpolant, interpolantT)

        tops = np.array([k.from_dist(grid_dists) for k in params.kernels])
        Bs = np.array(params.coreg_mats)
        bt = np.tensordot(Bs, tops, axes=(0, 0))
        blocked = _symm_2d_list_map(Toeplitz, bt, params.D)
        blocked = BlockMatrix(blocked)
        self.ski = SKI(blocked, interpolant, interpolantT)
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
