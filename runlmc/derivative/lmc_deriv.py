# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import itertools
import numpy as np

from .exact_deriv import ExactDeriv
from .stochastic_deriv import StochasticDeriv
from ..approx.ski import SKI
from ..linalg.toeplitz import Toeplitz
from ..linalg.kronecker import Kronecker
from ..linalg.sum_matrix import SumMatrix

# TODO(cleanup): document purpose: separated from paramz logic <- document
# TODO(cleanup): extract common functionality into the base class somehow?

class LMCDerivative:
    def coreg_vec_gradients(self):
        raise NotImplementedError

    def coreg_diag_gradients(self):
        raise NotImplementedError

    def kernel_gradients(self):
        raise NotImplementedError

    def noise_gradient(self):
        raise NotImplementedError

class ApproxLMCDerivative(LMCDerivative):
    # TODO(SLFM-representation)
    def __init__(self, coreg_vecs, coreg_diags, kernels,
                 dists, interpolant, interpolantT, lens, y, noise):
        self.coreg_vecs = coreg_vecs
        self.coreg_diag = coreg_diags
        self.kernels = kernels
        self.dists = dists
        self.coreg_mats = [np.outer(a, a) + np.diag(k)
                           for a, k in zip(coreg_vecs, self.coreg_diag)]
        self.materialized_kernels = [Toeplitz(k.from_dist(dists))
                                     for k in kernels]
        products = [Kronecker(A, K) for A, K in
                    zip(self.coreg_mats, self.materialized_kernels)]
        kern_sum = SumMatrix(products)
        self.interpolant = interpolant
        self.interpolantT = interpolantT
        self.n = len(y)
        self.D = len(self.coreg_diag[0])
        self.m = len(self.dists)
        self.noise = noise
        self.lens = lens
        # TODO(block-Toeplitz representation)
        self.ski = SKI(
            kern_sum,
            self.interpolant,
            self.interpolantT,
            np.repeat(noise, lens))
        self.deriv = StochasticDeriv(self.ski, y)

    def _ski_d(self, K):
        ski = SKI(K, self.interpolant, self.interpolantT, None)
        return self.deriv.derivative(ski)

    def coreg_vec_gradients(self):
        grads = []
        for a, toep_K in zip(self.coreg_vecs, self.materialized_kernels):
            grad = np.zeros(self.D)
            for i in range(self.D):
                A = np.zeros((self.D, self.D))
                A[i] += a
                A.T[i] += a
                # TODO(sparse-derivatives)
                dKdt = Kronecker(A, toep_K)
                grad[i] = self._ski_d(dKdt)
            grads.append(grad)
        return grads

    def coreg_diag_gradients(self):
        grads = []
        for toep_K in self.materialized_kernels:
            zeros = np.zeros((self.D, self.D))
            grad = np.zeros(self.D)
            for i in range(self.D):
                zeros[i, i] = 1
                # TODO(sparse-derivatives)
                dKdt = Kronecker(zeros, toep_K)
                grad[i] = self._ski_d(dKdt)
                zeros[i, i] = 0
            grads.append(grad)
        return grads

    def kernel_gradients(self):
        grads = []
        for A, kern in zip(self.coreg_mats, self.kernels):
            kern_grad = []
            for dKdt_toep in kern.kernel_gradient(self.dists):
                dKdt = Kronecker(A, Toeplitz(dKdt_toep))
                dLdt = self._ski_d(dKdt)
                kern_grad.append(dLdt)
            grads.append(kern_grad)
        return grads

    # TODO(sparse-derivatives) - move to linalg
    class _Diag:
        def __init__(self, v):
            self.v = v
        def matvec(self, x):
            return self.v * x

    def noise_gradient(self):
        grad = np.zeros(len(self.noise))
        for i in range(self.D):
            d_noise = np.zeros(self.D)
            d_noise[i] = 1
            d_noise = np.repeat(d_noise, self.lens)
            grad[i] = self.deriv.derivative(self._Diag(d_noise))
        return grad

class ExactLMCDerivative:

    def __init__(self, coreg_vecs, coreg_diags,
                 kernels, dists, lens, y, noise):
        self.coreg_vecs = coreg_vecs
        self.coreg_diag = coreg_diags
        self.kernels = kernels
        self.coreg_mats = [np.outer(a, a) + np.diag(k)
                           for a, k in zip(coreg_vecs, self.coreg_diag)]
        self.materialized_kernels = [k.from_dist(dists) for k in kernels]
        self.dists = dists
        self.n = len(y)
        self.D = len(self.coreg_diag[0])
        self.m = len(self.dists)
        self.noise = noise
        self.repeated_noise = np.repeat(noise, lens)
        self.lens = lens
        self.K = sum(self.coreg_scale(A, Kq) for A, Kq in
                     zip(self.coreg_mats, self.materialized_kernels))
        self.K += np.diag(self.repeated_noise)
        self.deriv = ExactDeriv(self.K, y)

    def coreg_scale(self, A, K):
        # TODO(cleanup): this should be a single method in np convenience
        ends = np.add.accumulate(self.lens)
        begins = np.roll(ends, 1)
        begins[0] = 0
        K = np.copy(K)
        for i, j in itertools.product(range(self.D), range(self.D)):
            rbegin, rend = begins[i], ends[i]
            cbegin, cend = begins[j], ends[j]
            K[rbegin:rend, cbegin:cend] *= A[i, j]
        return K

    def coreg_vec_gradients(self):
        grads = []
        for a, Kq in zip(self.coreg_vecs, self.materialized_kernels):
            grad = np.zeros(self.D)
            for i in range(self.D):
                A = np.zeros((self.D, self.D))
                A[i] += a
                A.T[i] += a
                dKdt = self.coreg_scale(A, Kq)
                grad[i] = self.deriv.derivative(dKdt)
            grads.append(grad)
        return grads

    def coreg_diag_gradients(self):
        grads = []
        for Kq in self.materialized_kernels:
            zeros = np.zeros((self.D, self.D))
            grad = np.zeros(self.D)
            for i in range(self.D):
                zeros[i, i] = 1
                dKdt = self.coreg_scale(zeros, Kq)
                grad[i] = self.deriv.derivative(dKdt)
                zeros[i, i] = 0
            grads.append(grad)
        return grads

    def kernel_gradients(self):
        grads = []
        for A, kern in zip(self.coreg_mats, self.kernels):
            kern_grad = []
            for dKdt in kern.kernel_gradient(self.dists):
                dKdt = self.coreg_scale(A, dKdt)
                dLdt = self.deriv.derivative(dKdt)
                kern_grad.append(dLdt)
            grads.append(kern_grad)
        return grads

    def noise_gradient(self):
        grad = np.zeros(len(self.noise))
        for i in range(self.D):
            d_noise = np.zeros(self.D)
            d_noise[i] = 1
            d_noise = np.repeat(d_noise, self.lens)
            grad[i] = self.deriv.derivative(np.diag(d_noise))
        return grad
