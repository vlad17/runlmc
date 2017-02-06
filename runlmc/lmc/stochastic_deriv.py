# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

from multiprocessing import Pool
from contextlib import closing

import numpy as np

from .derivative import Derivative
from ..approx.iterative import Iterative

# https://github.com/mauriziofilippone/preconditioned_GPs/blob/d7bc09b6804ef002cc3fc6bbf936517578d7436e/code/accuracy_vs_time/gp_functions/gp_regression_cg.r

class StochasticDeriv(Derivative):
    # This code accepts arbitrary linear operators for the derivatives
    # K, however, should have a "solve" function

    N_IT = 3

    def __init__(self, K, y, metrics, pool=None):
        self.n = K.shape[0]
        self.K = K

        self.rs = np.random.randint(0, 2, (self.N_IT, self.n)) * 2 - 1

        record_metrics = metrics is not None
        to_invert = [(K, y, record_metrics)] + [
            (K, x, record_metrics) for x in self.rs]

        if record_metrics:
            solved, ctrs, errs = zip(
                *StochasticDeriv._concurrent_solve(pool, to_invert))
            metrics.iterations.append(np.mean(ctrs))
            metrics.solv_error.append(np.mean(errs))
            solved = list(solved)
        else:
            solved = StochasticDeriv._concurrent_solve(pool, to_invert)

        self.alpha = solved[0]
        self.inv_rs = solved[1:]

    @staticmethod
    def _concurrent_solve(pool, ls):
        if pool is None:
            return [Iterative.solve(*x) for x in ls]
        else:
            return pool.starmap(Iterative.solve, ls)

    def d_normal_quadratic(self, dKdt):
        return self.alpha.dot(dKdt.matvec(self.alpha))

    def d_logdet_K(self, dKdt):
        # Preconditioning Kernel Matrices, Cutajar 2016
        trace = 0
        for r, rinv in zip(self.rs, self.inv_rs):
            dr = dKdt.matvec(r)
            trace += rinv.dot(dr)
        return trace / self.N_IT
