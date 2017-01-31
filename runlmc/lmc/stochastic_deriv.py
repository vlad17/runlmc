# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import numpy as np

from .derivative import Derivative
from ..approx.iterative import Iterative

# https://github.com/mauriziofilippone/preconditioned_GPs/blob/d7bc09b6804ef002cc3fc6bbf936517578d7436e/code/accuracy_vs_time/gp_functions/gp_regression_cg.r

class StochasticDeriv(Derivative):
    # This code accepts arbitrary linear operators for the derivatives
    # K, however, should have a "solve" function

    N_IT = 2

    def __init__(self, K, y):
        self.n = K.shape[0]
        self.K = K
        self.alpha = Iterative.solve(K, y)
        self.rs = np.random.randint(0, 2, (self.N_IT, self.n)) * 2 - 1
        self.inv_rs = [Iterative.solve(K, r) for r in self.rs]

    def d_normal_quadratic(self, dKdt):
        return self.alpha.dot(dKdt.matvec(self.alpha))

    def d_logdet_K(self, dKdt):
        # Preconditioning Kernel Matrices, Cutajar 2016
        trace = 0
        for r, rinv in zip(self.rs, self.inv_rs):
            dr = dKdt.matvec(r)
            trace += rinv.dot(dr)
        return trace / self.N_IT
