# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import numpy as np

from .derivative import Derivative
from ..approx.iterative import Iterative

# https://github.com/mauriziofilippone/preconditioned_GPs/blob/d7bc09b6804ef002cc3fc6bbf936517578d7436e/code/accuracy_vs_time/gp_functions/gp_regression_cg.r


class StochasticDerivService:
    """
    This service generates :class:`runlmc.lmc.Derivative` instances with
    pre-specified configurations for recording metrics or using
    multiprocessing, which enables decoupling of the math from the
    systems in the GP logic.

    :param metrics: a :class:`runlmc.lmc.metrics.Metrics` instance or `None`
                    (if no metrics are to be recorded)
    :param pool: pool for parallel processing
    :param n_it: iterations to use in stochastic trace approximation
    :param tol: tolerance in inversion routine
    :ivar metrics: the metrics instance used by this class
    """

    def __init__(self, metrics, pool, n_it, tol):
        self.metrics = metrics
        self._pool = pool
        self._n_it = n_it
        self._tol = tol

    def generate(self, K, y):
        n = K.shape[0]
        rs = np.random.randint(0, 2, (self._n_it, n)) * 2 - 1
        record_metrics = self.metrics is not None
        minres = True
        tol = self._tol
        to_invert = [(K, y, record_metrics, minres, tol)] + [
            (K, x, record_metrics, minres, tol) for x in rs]

        if record_metrics:
            solved, ctrs, errs = zip(*self._concurrent_solve(to_invert))
            self.metrics.iterations.append(np.mean(ctrs))
            self.metrics.solv_error.append(np.mean(errs))
        else:
            solved = self._concurrent_solve(to_invert)

        return StochasticDeriv(solved[0], rs, solved[1:], self._n_it)

    def _concurrent_solve(self, ls):
        return self._pool.starmap(Iterative.solve, ls)


class StochasticDeriv(Derivative):
    """
    Given the inverse of random binary vectors `inv_rs` with respect to some
    kernel :math:`K` and the similar inverse `alpha` of observations
    :math:`K^{-1}y`, this class produces the derivatives of :math:`K` with
    respect to its hyperparameters.
    """

    def __init__(self, alpha, rs, inv_rs, n_it):
        self.alpha = alpha
        self._rs = rs
        self._inv_rs = inv_rs
        self._n_it = n_it

    def d_normal_quadratic(self, dKdt):
        return self.alpha.dot(dKdt.matvec(self.alpha))

    def d_logdet_K(self, dKdt):
        # Preconditioning Kernel Matrices, Cutajar 2016
        trace = 0
        for r, rinv in zip(self._rs, self._inv_rs):
            dr = dKdt.matvec(r)
            trace += rinv.dot(dr)
        return trace / self._n_it
