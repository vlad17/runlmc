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
                 interpolant, interpolantT, ktype, active_dim):
        n = interpolant.shape[0]
        super().__init__(n, n)
        grid_k = functional_kernel.eval_kernels_fixed_dim(
            grid_dists, active_dim)

        if ktype == 'sum':
            self.grid_K = _gen_sum_grid(functional_kernel, grid_k, active_dim)
        elif ktype == 'bt':
            self.grid_K = _gen_bt_grid(functional_kernel, grid_k, active_dim)
        elif ktype == 'slfm':
            self.grid_K = _gen_slfm_grid(
                functional_kernel, grid_k, n, active_dim)
        else:
            assert False, ktype

        self.ski = SKI(self.grid_K, interpolant, interpolantT)

    def matvec(self, x):
        return self.ski.matvec(x)

# TODO(test)


def gen_grid_kernel(fk, grid_dists, interpolants, lens_per_output):
    all_gk = []
    for active_dim in fk.active_dims.keys():
        if fk.Q == 1:
            ktype = 'sum'
        else:
            tot_rank = fk.total_rank(active_dim)
            if not fk.num_lmc[active_dim] and not fk.num_indep[active_dim]:
                correction_if_no_diagonal = fk.D
            else:
                correction_if_no_diagonal = 0
            dsq = fk.D ** 2
            if tot_rank + fk.D < dsq + correction_if_no_diagonal:
                ktype = 'slfm'
            else:
                ktype = 'bt'
        grid_dist = grid_dists[active_dim]
        interpolant, interpolantT = interpolants[active_dim]
        gk = GridKernel(fk, grid_dist, interpolant,
                        interpolantT, ktype, active_dim)
        all_gk.append(gk)

    noise = Diag(np.repeat(fk.noise, lens_per_output))
    all_gk.append(noise)

    return SumMatrix(all_gk)


def _gen_slfm_grid(functional_kernel, grid_k, n, active_dim):
    coreg_Ks = _gen_coreg_Ks(functional_kernel, grid_k, active_dim)
    diag_Ks = _gen_diag_Ks(functional_kernel, grid_k, n, active_dim)
    return SumMatrix([coreg_Ks, diag_Ks])


def _gen_coreg_Ks(fk, grid_k, active_dim):
    kidxs = fk.active_dims[active_dim]
    non_indep_max = fk.num_lmc[active_dim] + fk.num_slfm[active_dim]
    non_indep = [kidx for kidx in kidxs if kidx < non_indep_max]
    all_coreg = [fk.coreg_vecs[idx] for idx in non_indep]
    ranks = np.array([len(coreg) for coreg in all_coreg])
    A_star = np.vstack(all_coreg).T
    I_m = Identity(np.prod(grid_k.shape[1:]))
    left = Kronecker(NumpyMatrix(A_star), I_m)
    right = Kronecker(NumpyMatrix(A_star.T), I_m)
    # here, rely on the fact that grid_k from fk.evaluate_kernels_fixed_dim
    # is returned in the same order as fk.active_dims[active_dim]
    deduped_toeps = np.array([BTTB(top.ravel(), top.shape)
                              for top in grid_k[:len(all_coreg)]])
    toeps = BlockDiag(np.repeat(deduped_toeps, ranks))
    coreg_Ks = Composition([left, toeps, right])
    return coreg_Ks


def _gen_diag_Ks(fk, grid_k, n, active_dim):
    if fk.num_lmc[active_dim] == 0 and fk.num_indep[active_dim] == 0:
        return Identity(n)
    kidxs = fk.active_dims[active_dim]
    diags = np.column_stack([fk.coreg_diags[kidx] for kidx in kidxs])
    Q = grid_k.shape[0]
    assert Q == len(kidxs)
    diag_tops = diags.dot(grid_k.reshape(Q, -1))
    diag_Ks = BlockDiag([BTTB(top, grid_k.shape[1:]) for top in diag_tops])
    return diag_Ks


def _gen_bt_grid(functional_kernel, grid_k, active_dim):
    Bs = np.array(functional_kernel.coreg_mats(active_dim))
    Q = grid_k.shape[0]
    tops = grid_k.reshape(Q, -1)
    bt = np.tensordot(Bs, tops, axes=(0, 0))
    sizes = grid_k.shape[1:]
    blocked = symm_2d_list_map(BTTB, bt, functional_kernel.D, sizes)
    blocked = SymmSquareBlockMatrix(blocked)
    return blocked


def _gen_sum_grid(functional_kernel, grid_k, active_dim):
    Q = grid_k.shape[0]
    tops = grid_k.reshape(Q, -1)
    sizes = grid_k.shape[1:]
    kernels_on_grid = [BTTB(top, sizes) for top in tops]
    # TODO(sum-fast)
    # Coreg_mats can be in decomposed representation to be a bit faster.
    products = [Kronecker(NumpyMatrix(A), K) for A, K in
                zip(functional_kernel.coreg_mats(active_dim), kernels_on_grid)]
    ksum = SumMatrix(products)
    return ksum
