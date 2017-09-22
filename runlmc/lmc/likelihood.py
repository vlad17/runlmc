# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import itertools
import numpy as np
import scipy.linalg as la
import scipy.spatial.distance as dist

from .exact_deriv import ExactDeriv
from ..approx.ski import SKI
from ..linalg.diag import Diag
from ..linalg.toeplitz import Toeplitz
from ..linalg.kronecker import Kronecker
from ..linalg.numpy_matrix import NumpyMatrix
from ..util.numpy_convenience import begin_end_indices

# TODO(test): all of below


class LMCLikelihood:
    """
    Separate hyperparameter-based likelihood differentiation from the
    model class for separation of concerns. Different sub-classes may implement
    the below methods differently, with different asymptotic performance
    properties.
    """

    def __init__(self, functional_kernel, Ys):
        self.functional_kernel = functional_kernel
        self.y = np.hstack(Ys)
        self.lens = list(map(len, Ys))

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
        for q, a in enumerate(self.functional_kernel.coreg_vecs):
            grad = np.zeros(a.shape)
            for i, ai in enumerate(a):
                for j in range(self.functional_kernel.D):
                    dAdt = np.zeros((self.functional_kernel.D,
                                     self.functional_kernel.D))
                    dAdt[j] += ai
                    dAdt.T[j] += ai
                    # TODO(sparse-derivatives)
                    dKdt = self._dKdt_from_dAdt(dAdt, q)
                    grad[i, j] = self._dLdt_from_dKdt(dKdt)
            grads.append(grad)
        return grads

    def coreg_diags_gradients(self):
        grads = []
        for q in range(self.functional_kernel.Q):
            zeros = np.zeros(
                (self.functional_kernel.D, self.functional_kernel.D))
            grad = np.zeros(self.functional_kernel.D)
            for i in range(self.functional_kernel.D):
                zeros[i, i] = 1
                # TODO(sparse-derivatives)
                dKdt = self._dKdt_from_dAdt(zeros, q)
                grad[i] = self._dLdt_from_dKdt(dKdt)
                zeros[i, i] = 0
            grads.append(grad)
        return grads

    def kernel_gradients(self):
        grads = []
        for q, A in enumerate(self.functional_kernel.coreg_mats()):
            kern_grad = []
            for dKdt in self._dKdts_from_dKqdts(A, q):
                dLdt = self._dLdt_from_dKdt(dKdt)
                kern_grad.append(dLdt)
            grads.append(kern_grad)
        return grads

    def noise_gradient(self):
        grad = np.zeros(len(self.functional_kernel.noise))
        for i in range(self.functional_kernel.D):
            d_noise = np.zeros(self.functional_kernel.D)
            d_noise[i] = 1
            dKdt = self._dKdt_from_dEpsdt(d_noise)
            grad[i] = self._dLdt_from_dKdt(dKdt)
        return grad


class ApproxLMCLikelihood(LMCLikelihood):
    def __init__(self, functional_kernel, grid_kern, grid_dists, Ys, deriv):
        super().__init__(functional_kernel, Ys)
        kernels_on_grid = self.functional_kernel.eval_kernels(grid_dists)
        self.materialized_kernels = [Toeplitz(d) for d in kernels_on_grid]
        self.K = grid_kern
        self.deriv = deriv.generate(self.K, self.y)
        self.materialized_grads = self.functional_kernel.eval_kernel_gradients(
            grid_dists)

    def _ski(self, X):
        return SKI(X, *self.K.interpolants())

    def _dKdt_from_dAdt(self, dAdt, q):
        return self._ski(Kronecker(
            NumpyMatrix(dAdt), self.materialized_kernels[q]))

    def _dKdts_from_dKqdts(self, A, q):
        for dKqdt in self.materialized_grads[q]:
            yield self._ski(Kronecker(
                NumpyMatrix(A), Toeplitz(dKqdt)))

    def _dKdt_from_dEpsdt(self, dEpsdt):
        # no SKI approximation for noise
        return Diag(np.repeat(dEpsdt, self.lens))

    def _dLdt_from_dKdt(self, dKdt):
        return self.deriv.derivative(dKdt)

    def alpha(self):
        return self.deriv.alpha


class ExactLMCLikelihood(LMCLikelihood):
    def __init__(self, functional_kernel, Xs, Ys):
        super().__init__(functional_kernel, Ys)

        # TODO(1d)
        pdists = dist.pdist(np.hstack(Xs).reshape(-1, 1))
        pdists = dist.squareform(pdists)

        self.materialized_kernels = self.functional_kernel.eval_kernels(pdists)
        self.K = sum(self._personalized_coreg_scale(A, Kq) for A, Kq in
                     zip(self.functional_kernel.coreg_mats(),
                         self.materialized_kernels))
        self.K += np.diag(np.repeat(functional_kernel.noise, self.lens))
        self.L = la.cho_factor(self.K)
        self.deriv = ExactDeriv(self.L, self.y)
        self.materialized_grads = self.functional_kernel.eval_kernel_gradients(
            pdists)

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
        return ExactLMCLikelihood._coreg_scale(
            A, K, self.lens, self.lens, self.functional_kernel.D)

    @staticmethod
    def kernel_from_indices(Xs, Zs, functional_kernel):
        """Computes the dense, exact kernel matrix for an LMC kernel specified
        by `functional_kernel`. The kernel matrix that is computed is relative
        to the kernel application to pairs from the Cartesian product `Xs` and
        `Zs`.
        """

        # TODO(1d)
        pair_dists = dist.cdist(np.hstack(Xs).reshape(-1, 1),
                                np.hstack(Zs).reshape(-1, 1))
        Kqs = functional_kernel.eval_kernels(pair_dists)
        rlens, clens = [len(X) for X in Xs], [len(Z) for Z in Zs]
        K = sum(ExactLMCLikelihood._coreg_scale(A, Kq, rlens, clens,
                                                functional_kernel.D)
                for A, Kq in zip(functional_kernel.coreg_mats(), Kqs))
        return K

    def _dKdt_from_dAdt(self, dAdt, q):
        return self._personalized_coreg_scale(
            dAdt, self.materialized_kernels[q])

    def _dKdts_from_dKqdts(self, A, q):
        for dKqdt in self.materialized_grads[q]:
            yield self._personalized_coreg_scale(A, dKqdt)

    def _dKdt_from_dEpsdt(self, dEpsdt):
        return np.diag(np.repeat(dEpsdt, self.lens))

    def _dLdt_from_dKdt(self, dKdt):
        return self.deriv.derivative(dKdt)

    def alpha(self):
        return self.deriv.alpha
