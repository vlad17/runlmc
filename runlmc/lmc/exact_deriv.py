# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import numpy as np
import scipy.linalg as la

from .derivative import Derivative

class ExactDeriv(Derivative):
    # GPML footnote on p.114
    # This code assumes dKdt is dense

    def __init__(self, L, y):
        self.n = L[0].shape[0]
        self.L = L
        self.Kinv = la.cho_solve(self.L, np.identity(self.n), overwrite_b=True)
        self.alpha = la.cho_solve(self.L, y, overwrite_b=False)

    def d_normal_quadratic(self, dKdt):
        return self.alpha.dot(dKdt.dot(self.alpha))

    def d_logdet_K(self, dKdt):
        return (dKdt * self.Kinv).sum()
