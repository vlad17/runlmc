# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import itertools
import numpy as np
import scipy.linalg as la

from .exact_deriv import ExactDeriv
from .stochastic_deriv import StochasticDeriv
from ..approx.ski import SKI
from ..linalg.toeplitz import Toeplitz
from ..linalg.kronecker import Kronecker

# TODO(cleanup): document purpose: separated from paramz logic <- document

class LMCKernel:

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

    def coreg_vec_gradients(self):
        grads = []
        for q, a in enumerate(self.params.coreg_vecs):
            grad = np.zeros(self.params.D)
            for i in range(self.params.D):
                dAdt = np.zeros((self.params.D, self.params.D))
                dAdt[i] += a
                dAdt.T[i] += a
                # TODO(sparse-derivatives)
                dKdt = self._dKdt_from_dAdt(dAdt, q)
                grad[i] = self._dLdt_from_dKdt(dKdt)
            grads.append(grad)
        return grads

    def coreg_diag_gradients(self):
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
    def __init__(self, grid_kern):
        super().__init__(grid_kern.params)
        self.materialized_kernels = [Toeplitz(k.from_dist(grid_kern.dists))
                                     for k in self.params.kernels]
        self.interpolant = grid_kern.interpolant
        self.interpolantT = grid_kern.interpolantT
        self.K = grid_kern
        self.deriv = StochasticDeriv(self.K, self.params.y)

    def _ski(self, X):
        return SKI(X, self.interpolant, self.interpolantT)

    def _dKdt_from_dAdt(self, dAdt, q):
        return self._ski(Kronecker(dAdt, self.materialized_kernels[q]))

    def _dKdts_from_dKqdts(self, A, q):
        for dKqdt in self.params.kernels[q].kernel_gradient(self.K.dists):
            yield self._ski(Kronecker(A, Toeplitz(dKqdt)))

    # TODO(sparse-derivatives) - move to linalg
    class _Diag:
        def __init__(self, v):
            self.v = v
        def matvec(self, x):
            return self.v * x

    def _dKdt_from_dEpsdt(self, dEpsdt):
        # no SKI approximation necessary for noise
        return ApproxLMCKernel._Diag(np.repeat(dEpsdt, self.params.lens))

    def _dLdt_from_dKdt(self, dKdt):
        return self.deriv.derivative(dKdt)


class ExactLMCKernel(LMCKernel):
    def __init__(self, params, pair_dists):
        super().__init__(params)
        self.materialized_kernels = [k.from_dist(pair_dists)
                                     for k in params.kernels]
        self.pair_dists = pair_dists
        self.K = sum(self.coreg_scale(A, Kq) for A, Kq in
                     zip(params.coreg_mats, self.materialized_kernels))
        self.K += np.diag(np.repeat(params.noise, params.lens))
        self.L = la.cho_factor(self.K)
        self.deriv = ExactDeriv(self.L, self.params.y)

    def coreg_scale(self, A, K):
        # TODO(cleanup): this should be a single method in np convenience
        ends = np.add.accumulate(self.params.lens)
        begins = np.roll(ends, 1)
        begins[0] = 0
        K = np.copy(K)
        D = self.params.D
        for i, j in itertools.product(range(D), range(D)):
            rbegin, rend = begins[i], ends[i]
            cbegin, cend = begins[j], ends[j]
            K[rbegin:rend, cbegin:cend] *= A[i, j]
        return K

    def _dKdt_from_dAdt(self, dAdt, q):
        return self.coreg_scale(dAdt, self.materialized_kernels[q])

    def _dKdts_from_dKqdts(self, A, q):
        for dKqdt in self.params.kernels[q].kernel_gradient(self.pair_dists):
            yield self.coreg_scale(A, dKqdt)

    def _dKdt_from_dEpsdt(self, dEpsdt):
        return np.diag(np.repeat(dEpsdt, self.params.lens))

    def _dLdt_from_dKdt(self, dKdt):
        return self.deriv.derivative(dKdt)
