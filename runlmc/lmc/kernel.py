# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import itertools
import numpy as np
import scipy.linalg as la
import scipy.spatial.distance as dist

from .exact_deriv import ExactDeriv
from .stochastic_deriv import StochasticDeriv
from ..approx.ski import SKI
from ..linalg.diag import Diag
from ..linalg.toeplitz import Toeplitz
from ..linalg.kronecker import Kronecker
from ..linalg.numpy_matrix import NumpyMatrix
from ..util.numpy_convenience import begin_end_indices

# TODO(test): all of below


class LMCKernel:
    """
    Separate hyperparameter-based likelihood differentiation from the
    model class for separation of concerns. Different sub-classes may implement
    the below methods differently, with different asymptotic performance
    properties.
    """

    def __init__(self, params):
        self.params = params

    def _dKdt_from_dAdt(self, dAdt, q):
        raise NotImplementedError

    def _dKdts_from_dKqdts(self, A, q):
        raise NotImplementedError

    def _dKdt_from_dEpsdt(self, dEpsdt):
        raise NotImplementedError

    def _dLdt_from_dKdt(self, dKdt):
        raise NotImplementedError

    def alpha(self):
        raise NotImplementedError

    def coreg_vec_gradients(self):
        grads = []
        for q, a in enumerate(self.params.coreg_vecs):
            grad = np.zeros(a.shape)
            for i, ai in enumerate(a):
                for j in range(self.params.D):
                    dAdt = np.zeros((self.params.D, self.params.D))
                    dAdt[j] += ai
                    dAdt.T[j] += ai
                    # TODO(sparse-derivatives)
                    dKdt = self._dKdt_from_dAdt(dAdt, q)
                    grad[i, j] = self._dLdt_from_dKdt(dKdt)
            grads.append(grad)
        return grads

    def coreg_diags_gradients(self):
        grads = []
        for q in range(self.params.Q):
            zeros = np.zeros((self.params.D, self.params.D))
            grad = np.zeros(self.params.D)
            for i in range(self.params.D):
                zeros[i, i] = 1
                # TODO(sparse-derivatives)
                dKdt = self._dKdt_from_dAdt(zeros, q)
                grad[i] = self._dLdt_from_dKdt(dKdt)
                zeros[i, i] = 0
            grads.append(grad)
        return grads

    def kernel_gradients(self):
        grads = []
        for q, A in enumerate(self.params.coreg_mats):
            kern_grad = []
            for dKdt in self._dKdts_from_dKqdts(A, q):
                dLdt = self._dLdt_from_dKdt(dKdt)
                kern_grad.append(dLdt)
            grads.append(kern_grad)
        return grads

    def noise_gradient(self):
        grad = np.zeros(len(self.params.noise))
        for i in range(self.params.D):
            d_noise = np.zeros(self.params.D)
            d_noise[i] = 1
            dKdt = self._dKdt_from_dEpsdt(d_noise)
            grad[i] = self._dLdt_from_dKdt(dKdt)
        return grad


class ApproxLMCKernel(LMCKernel):
    def __init__(self, params, grid_kern, grid_dists, metrics, pool=None):
        super().__init__(params)
        self.materialized_kernels = [Toeplitz(k.from_dist(grid_dists))
                                     for k in self.params.kernels]
        self.K = grid_kern
        self.dists = grid_dists
        self.deriv = StochasticDeriv(self.K, self.params.y, metrics, pool)

    def _ski(self, X):
        return SKI(X, *self.K.interpolants())

    def _dKdt_from_dAdt(self, dAdt, q):
        return self._ski(Kronecker(
            NumpyMatrix(dAdt), self.materialized_kernels[q]))

    def _dKdts_from_dKqdts(self, A, q):
        for dKqdt in self.params.kernels[q].kernel_gradient(self.dists):
            yield self._ski(Kronecker(
                NumpyMatrix(A), Toeplitz(dKqdt)))

    def _dKdt_from_dEpsdt(self, dEpsdt):
        # no SKI approximation for noise
        return Diag(np.repeat(dEpsdt, self.params.lens))

    def _dLdt_from_dKdt(self, dKdt):
        return self.deriv.derivative(dKdt)

    def alpha(self):
        return self.deriv.alpha


class ExactLMCKernel(LMCKernel):
    def __init__(self, params, Xs):
        super().__init__(params)

        # TODO(1d)
        pdists = dist.pdist(np.hstack(Xs).reshape(-1, 1))
        pdists = dist.squareform(pdists)

        self.materialized_kernels = [
            k.from_dist(pdists) for k in params.kernels]
        self.pair_dists = pdists
        self.K = sum(self._personalized_coreg_scale(A, Kq) for A, Kq in
                     zip(params.coreg_mats, self.materialized_kernels))
        self.K += np.diag(np.repeat(params.noise, params.lens))
        self.L = la.cho_factor(self.K)
        self.deriv = ExactDeriv(self.L, self.params.y)

    @staticmethod
    def _coreg_scale(A, K, row_block_lens, col_block_lens, D):
        rbegins, rends = begin_end_indices(row_block_lens)
        cbegins, cends = begin_end_indices(col_block_lens)
        K = np.copy(K)
        for i, j in itertools.product(range(D), range(D)):
            rbegin, rend = rbegins[i], rends[i]
            cbegin, cend = cbegins[j], cends[j]
            K[rbegin:rend, cbegin:cend] *= A[i, j]
        return K

    def _personalized_coreg_scale(self, A, K):
        return ExactLMCKernel._coreg_scale(
            A, K, self.params.lens, self.params.lens, self.params.D)

    @staticmethod
    def from_indices(Xs, Zs, params):
        """Computes the dense, exact kernel matrix for an LMC kernel specified
        by `params`. The kernel matrix that is computed is relative to the
        kernel application to pairs from the Cartesian product `Xs` and `Zs`.

        This means that `params.y` and `params.lens` are unused."""

        #TODO(1d)
        pair_dists = dist.cdist(np.hstack(Xs).reshape(-1, 1), np.hstack(Zs).reshape(-1, 1))
        Kqs = [k.from_dist(pair_dists) for k in params.kernels]
        rlens, clens = [len(X) for X in Xs], [len(Z) for Z in Zs]
        K = sum(ExactLMCKernel._coreg_scale(A, Kq, rlens, clens, params.D)
                for A, Kq in zip(params.coreg_mats, Kqs))
        return K

    def _dKdt_from_dAdt(self, dAdt, q):
        return self._personalized_coreg_scale(
            dAdt, self.materialized_kernels[q])

    def _dKdts_from_dKqdts(self, A, q):
        for dKqdt in self.params.kernels[q].kernel_gradient(self.pair_dists):
            yield self._personalized_coreg_scale(A, dKqdt)

    def _dKdt_from_dEpsdt(self, dEpsdt):
        return np.diag(np.repeat(dEpsdt, self.params.lens))

    def _dLdt_from_dKdt(self, dKdt):
        return self.deriv.derivative(dKdt)

    def alpha(self):
        return self.deriv.alpha
