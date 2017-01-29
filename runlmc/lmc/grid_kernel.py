# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import numpy as np

from ..approx.ski import SKI
from ..linalg.block_matrix import BlockMatrix
from ..linalg.symmetric_matrix import SymmetricMatrix
from ..linalg.toeplitz import Toeplitz
from ..linalg.kronecker import Kronecker
from ..linalg.sum_matrix import SumMatrix
from ..util.docs import inherit_doc
from ..util.numpy_convenience import symm_2d_list_map

# TODO(test)
class GridKernel(SymmetricMatrix):
    def __init__(self, params, grid_dists, interpolant, interpolantT):
        super().__init__(params.n)
        self.params = params
        self.dists = grid_dists
        self.interpolant = interpolant
        self.interpolantT = interpolantT

    def grid_only(self):
        raise NotImplementedError

# TODO(test)
def gen_grid_kernel(params, grid_dists, interpolant, interpolantT):
    if params.Q > 2 * params.D:
        ktype = BlockToeplitz
    else:
        ktype = SumGrid
    return ktype(params, grid_dists, interpolant, interpolantT)

# TODO(test)
@inherit_doc
class BlockToeplitz(GridKernel):
    def __init__(self, params, grid_dists, interpolant, interpolantT):
        super().__init__(params, grid_dists, interpolant, interpolantT)

        tops = np.array([k.from_dist(grid_dists) for k in params.kernels])
        Bs = np.array(params.coreg_mats)
        bt = np.tensordot(Bs, tops, axes=(0, 0))
        blocked = symm_2d_list_map(Toeplitz, bt, params.D)
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

# TODO(test)
@inherit_doc
class SumGrid(GridKernel):
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
