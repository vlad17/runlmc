# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import numpy as np

from ..approx.ski import SKI
from ..linalg.block_diag import BlockDiag
from ..linalg.block_matrix import SymmSquareBlockMatrix
from ..linalg.matrix import Matrix
from ..linalg.composition import Composition
from ..linalg.toeplitz import Toeplitz
from ..linalg.identity import Identity
from ..linalg.kronecker import Kronecker
from ..linalg.numpy_matrix import NumpyMatrix
from ..linalg.sum_matrix import SumMatrix
from ..util.docs import inherit_doc
from ..util.numpy_convenience import symm_2d_list_map

# TODO(test)
class GridKernel(Matrix):
    def __init__(self, params, grid_dists, interpolant, interpolantT):
        super().__init__(params.n, params.n)
        self.params = params
        self.dists = grid_dists
        self.interpolant = interpolant
        self.interpolantT = interpolantT

    def grid_only(self):
        raise NotImplementedError

# TODO(cleanup) can definately refactor below replication: they're
# all just summatrix of diag

# TODO(test)
def gen_grid_kernel(params, grid_dists, interpolant, interpolantT):
    if params.Q > 2 * params.D:
        ktype = BlockToeplitz
    else:
        ktype = SumGrid
    return ktype(params, grid_dists, interpolant, interpolantT)

# TODO(test)
class GridSLFM(GridKernel):
    def __init__(self, params, grid_dists, interpolant, interpolantT):
        super().__init__(params, grid_dists, interpolant, interpolantT)

        tops = np.array([k.from_dist(grid_dists) for k in params.kernels])
        coreg_Ks = GridSLFM._gen_coreg_Ks(params, tops)
        diag_Ks = GridSLFM._gen_diag_Ks(params, tops)

        self.ski = SKI(SumMatrix([coreg_Ks, diag_Ks]),
                       interpolant, interpolantT)
        self.noise = np.repeat(params.noise, params.lens)

    @staticmethod
    def _gen_coreg_Ks(params, tops):
        ranks = np.array([len(coreg) for coreg in params.coreg_vecs])
        A_star = np.vstack(params.coreg_vecs).T
        I_m = Identity(tops.shape[1])
        left = Kronecker(NumpyMatrix(A_star), I_m)
        right = Kronecker(NumpyMatrix(A_star.T), I_m)
        deduped_toeps = [Toeplitz(top) for top in tops]
        toeps = np.repeat(deduped_toeps, ranks)
        coreg_Ks = Composition([left, BlockDiag(toeps), right])
        return coreg_Ks

    @staticmethod
    def _gen_diag_Ks(params, tops):
        diags = np.column_stack(params.coreg_diag)
        diag_tops = diags.dot(tops)
        diag_Ks = BlockDiag([Toeplitz(top) for top in diag_tops])
        return diag_Ks

    def matvec(self, x):
        return self.ski.matvec(x) + self.noise * x

    def grid_only(self):
        raise NotImplementedError

# Note in theory we can save on computation of D^2 Toeplitz
# circ_ffts in BlockToeplitz below by taking the Q FFTs first then
# adding them by linearity, but this only helps if Q < D^2 and also
# we have to make D^2 FFTs during matvec anyway.

# TODO(test)
@inherit_doc
class BlockToeplitz(GridKernel):
    def __init__(self, params, grid_dists, interpolant, interpolantT):
        super().__init__(params, grid_dists, interpolant, interpolantT)

        tops = np.array([k.from_dist(grid_dists) for k in params.kernels])
        Bs = np.array(params.coreg_mats)
        bt = np.tensordot(Bs, tops, axes=(0, 0))
        blocked = symm_2d_list_map(Toeplitz, bt, params.D)
        blocked = SymmSquareBlockMatrix(blocked)
        self.ski = SKI(blocked, interpolant, interpolantT)
        self.noise = np.repeat(params.noise, params.lens)

    def matvec(self, x):
        return self.ski.matvec(x) + self.noise * x

    def grid_only(self):
        return self.ski.K

# TODO(test)
@inherit_doc
class SumGrid(GridKernel):
    def __init__(self, params, grid_dists, interpolant, interpolantT):
        super().__init__(params, grid_dists, interpolant, interpolantT)
        kerns_on_grid = [Toeplitz(k.from_dist(grid_dists))
                         for k in params.kernels]
        # Coreg_mats can be in decomposed representation to be a bit faster.
        products = [Kronecker(NumpyMatrix(A), K) for A, K in
                    zip(params.coreg_mats, kerns_on_grid)]
        ksum = SumMatrix(products)
        self.ski = SKI(ksum, interpolant, interpolantT)
        self.noise = np.repeat(params.noise, params.lens)

    def matvec(self, x):
        return self.ski.matvec(x) + self.noise * x

    def grid_only(self):
        return self.ski.K
