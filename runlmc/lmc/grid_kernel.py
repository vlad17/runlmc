# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import numpy as np

from ..approx.ski import SKI
from ..linalg.diag import Diag
from ..linalg.block_diag import BlockDiag
from ..linalg.block_matrix import SymmSquareBlockMatrix
from ..linalg.matrix import Matrix
from ..linalg.composition import Composition
from ..linalg.bttb import BTTB
from ..linalg.identity import Identity
from ..linalg.kronecker import Kronecker
from ..linalg.numpy_matrix import NumpyMatrix
from ..linalg.sum_matrix import SumMatrix
from ..util.numpy_convenience import symm_2d_list_map

# TODO(test)


class GridKernel(Matrix):
    def __init__(self, functional_kernel, grid_dists,
                 interpolant, interpolantT, ktype, lens_per_output):
        n = interpolant.shape[0]
        super().__init__(n, n)

        grid_k = functional_kernel.eval_kernels(grid_dists)

        if ktype == 'sum':
            self.grid_K = _gen_sum_grid(functional_kernel, grid_k)
        elif ktype == 'bt':
            self.grid_K = _gen_bt_grid(functional_kernel, grid_k)
        elif ktype == 'slfm':
            self.grid_K = _gen_slfm_grid(functional_kernel, grid_k, n)
        else:
            assert False, ktype

        self.ski = SKI(self.grid_K, interpolant, interpolantT)
        self.noise = Diag(
            np.repeat(functional_kernel.noise, lens_per_output))
        self.K = SumMatrix([self.ski, self.noise])

    def matvec(self, x):
        return self.K.matvec(x)

    def interpolants(self):
        ski = self.K.Ks[0]
        return ski.W, ski.WT

# TODO(test)


def gen_grid_kernel(functional_kernel, grid_dists, interpolant, interpolantT,
                    lens_per_output):
    if functional_kernel.Q == 1:
        ktype = 'sum'
    else:
        tot_rank = functional_kernel.total_rank()
        if functional_kernel.num_lmc == 0 and functional_kernel.num_indep == 0:
            correction_if_no_diagonal = functional_kernel.D
        else:
            correction_if_no_diagonal = 0
        dsq = functional_kernel.D ** 2
        if tot_rank + functional_kernel.D < dsq + correction_if_no_diagonal:
            ktype = 'slfm'
        else:
            ktype = 'bt'
    return GridKernel(functional_kernel, grid_dists,
                      interpolant, interpolantT, ktype, lens_per_output)


def _gen_slfm_grid(functional_kernel, grid_k, n):
    coreg_Ks = _gen_coreg_Ks(functional_kernel, grid_k)
    diag_Ks = _gen_diag_Ks(functional_kernel, grid_k, n)
    return SumMatrix([coreg_Ks, diag_Ks])


def _gen_coreg_Ks(functional_kernel, grid_k):
    non_indep = functional_kernel.num_lmc + functional_kernel.num_slfm
    all_coreg = functional_kernel.coreg_vecs[:non_indep]
    ranks = np.array([len(coreg) for coreg in all_coreg])
    A_star = np.vstack(all_coreg).T
    I_m = Identity(np.prod(grid_k.shape[1:]))
    left = Kronecker(NumpyMatrix(A_star), I_m)
    right = Kronecker(NumpyMatrix(A_star.T), I_m)
    deduped_toeps = np.array([BTTB(top, top.shape)
                              for top in grid_k[:len(all_coreg)]])
    toeps = BlockDiag(np.repeat(deduped_toeps, ranks))
    coreg_Ks = Composition([left, toeps, right])
    return coreg_Ks


def _gen_diag_Ks(functional_kernel, grid_k, n):
    if functional_kernel.num_lmc == 0 and functional_kernel.num_indep == 0:
        return Identity(n)
    diags = np.column_stack(functional_kernel.coreg_diags)
    Q = grid_k.shape[0]
    diag_tops = diags.dot(grid_k.reshape(Q, -1))
    diag_Ks = BlockDiag([BTTB(top, grid_k.shape[1:]) for top in diag_tops])
    return diag_Ks


def _gen_bt_grid(functional_kernel, grid_k):
    Bs = np.array(functional_kernel.coreg_mats())
    Q = grid_k.shape[0]
    tops = grid_k.reshape(Q, -1)
    bt = np.tensordot(Bs, tops, axes=(0, 0))
    sizes = grid_k.shape[1:]
    blocked = symm_2d_list_map(BTTB, bt, functional_kernel.D, sizes)
    blocked = SymmSquareBlockMatrix(blocked)
    return blocked


def _gen_sum_grid(functional_kernel, grid_k):
    Q = grid_k.shape[0]
    tops = grid_k.reshape(Q, -1)
    sizes = grid_k.shape[1:]
    kernels_on_grid = [BTTB(top, sizes) for top in tops]
    # TODO(sum-fast)
    # Coreg_mats can be in decomposed representation to be a bit faster.
    products = [Kronecker(NumpyMatrix(A), K) for A, K in
                zip(functional_kernel.coreg_mats(), kernels_on_grid)]
    ksum = SumMatrix(products)
    return ksum
