# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import numpy as np

from ..approx.ski import SKI
from ..linalg.diag import Diag
from ..linalg.block_diag import BlockDiag
from ..linalg.block_matrix import SymmSquareBlockMatrix
from ..linalg.matrix import Matrix
from ..linalg.composition import Composition
from ..linalg.toeplitz import Toeplitz
from ..linalg.identity import Identity
from ..linalg.kronecker import Kronecker
from ..linalg.numpy_matrix import NumpyMatrix
from ..linalg.sum_matrix import SumMatrix
from ..util.numpy_convenience import symm_2d_list_map

# TODO(test)
# notably for parallelism, this is paramz.Param-free.
class GridKernel(Matrix):
    def __init__(
            self, params, grid_dists, interpolant, interpolantT, ktype):
        super().__init__(params.n, params.n)

        tops = np.array([k.from_dist(grid_dists) for k in params.kernels])

        if ktype == 'sum':
            self.grid_K = _gen_sum_grid(params, tops)
        elif ktype == 'bt':
            self.grid_K = _gen_bt_grid(params, tops)
        elif ktype == 'slfm':
            self.grid_K = _gen_slfm_grid(params, tops)
        else:
            assert False, ktype

        self.ski = SKI(self.grid_K, interpolant, interpolantT)
        self.noise = Diag(np.repeat(params.noise, params.lens))
        self.K = SumMatrix([self.ski, self.noise])

    def matvec(self, x):
        return self.K.matvec(x)

    def interpolants(self):
        ski = self.K.Ks[0]
        return ski.W, ski.WT

# TODO(test)
def gen_grid_kernel(params, grid_dists, interpolant, interpolantT):
    if params.Q == 1:
        ktype = 'sum'
    else:
        tot_rank = sum(len(coreg) for coreg in params.coreg_vecs)
        no_diag_correction = (
            params.D if
            params.nkernels['lmc'] == 0 and params.nkernels['indep'] == 0 else
            0)
        dsq = params.D ** 2
        if tot_rank + params.D - no_diag_correction < dsq:
            ktype = 'slfm'
        else:
            ktype = 'bt'
    return GridKernel(params, grid_dists, interpolant, interpolantT, ktype)

def _gen_slfm_grid(params, tops):
    coreg_Ks = _gen_coreg_Ks(params, tops)
    diag_Ks = _gen_diag_Ks(params, tops)
    return SumMatrix([coreg_Ks, diag_Ks])

def _gen_coreg_Ks(params, tops):
    non_indep = params.nkernels['lmc'] + params.nkernels['slfm']
    all_coreg = params.coreg_vecs[:non_indep]
    ranks = np.array([len(coreg) for coreg in all_coreg])
    A_star = np.vstack(all_coreg).T
    I_m = Identity(tops.shape[1])
    left = Kronecker(NumpyMatrix(A_star), I_m)
    right = Kronecker(NumpyMatrix(A_star.T), I_m)
    deduped_toeps = np.array([Toeplitz(top) for top in tops[:len(all_coreg)]])
    toeps = BlockDiag(np.repeat(deduped_toeps, ranks))
    coreg_Ks = Composition([left, toeps, right])
    return coreg_Ks

def _gen_diag_Ks(params, tops):
    if params.nkernels['lmc'] == 0 and params.nkernels['indep'] == 0:
        return Identity(params.n)
    diags = np.column_stack(params.coreg_diags)
    diag_tops = diags.dot(tops)
    diag_Ks = BlockDiag([Toeplitz(top) for top in diag_tops])
    return diag_Ks

def _gen_bt_grid(params, tops):
    Bs = np.array(params.coreg_mats)
    bt = np.tensordot(Bs, tops, axes=(0, 0))
    blocked = symm_2d_list_map(Toeplitz, bt, params.D)
    blocked = SymmSquareBlockMatrix(blocked)
    return blocked

def _gen_sum_grid(params, tops):
    kernels_on_grid = [Toeplitz(top) for top in tops]
    # TODO(sum-fast)
    # Coreg_mats can be in decomposed representation to be a bit faster.
    products = [Kronecker(NumpyMatrix(A), K) for A, K in
                zip(params.coreg_mats, kernels_on_grid)]
    ksum = SumMatrix(products)
    return ksum
