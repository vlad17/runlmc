# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import numpy as np

from ..approx.ski import SKI
from ..linalg.block_diag import BlockDiag
from ..linalg.block_matrix import BlockMatrix
from ..linalg.symmetric_matrix import SymmetricMatrix
from ..linalg.toeplitz import Toeplitz
from ..linalg.identity import Identity
from ..linalg.kronecker import Kronecker, kron_mvprod
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

class _np:
    # TODO(cleanup) - just needs NumpyMatrix to be square
    def __init__(self, x):
        self.x = x
        self.shape = x.shape


    def matmat(self, x):
        return self.x.dot(x)

# TODO(test)
class GridSLFM(GridKernel):
    def __init__(self, params, grid_dists, interpolant, interpolantT):
        super().__init__(params, grid_dists, interpolant, interpolantT)

        # when rank > 1 need to decompose
        self.A_star = np.array(params.coreg_vecs).T
        self.I_m = Identity(len(grid_dists))
        tops = np.array([k.from_dist(grid_dists) for k in params.kernels])
        self.Ks = BlockDiag([Toeplitz(top) for top in tops])
        diags = np.array(params.coreg_diag).T
        diag_tops = diags.dot(tops)
        self.diag_Ks = BlockDiag([Toeplitz(top) for top in diag_tops])

        self.noise = np.repeat(params.noise, params.lens)


    def matvec(self, x):
        Wtx = self.interpolantT.dot(x)
        coreg_x = kron_mvprod(_np(self.A_star.T), self.I_m, Wtx) #not in-place!
        coreg_x = self.Ks.matvec(coreg_x)
        coreg_x = kron_mvprod(_np(self.A_star), self.I_m, coreg_x)
        diag_x = self.diag_Ks.matvec(Wtx)
        Wx = self.interpolant.dot(coreg_x + diag_x)
        return Wx + self.noise * x

    def as_numpy(self):
        ai = np.kron(self.A_star, self.I_m.as_numpy())
        center = ai.dot(self.Ks.matmat(ai.T))
        center += self.diag_Ks.as_numpy()
        centerWT = self.interpolant.dot(center).T
        x = self.interpolant.dot(centerWT)
        return x + np.diag(self.noise)

    def upper_eig_bound(self):
        raise NotImplementedError

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
        # Coreg_mats can be in decomposed representation to be a bit faster.
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
