# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import unittest

import numpy as np

from .lmc import LMC
from ..kern.rbf import RBF

class LMCTest(unittest.TestCase):

    def setUp(self):
        super().setUp()

    def test_no_kernel(self):
        mapnp = lambda x: list(map(np.array, x))
        basic_Xs = mapnp([[0, 1, 2], [0.5, 1.5, 2.5]])
        basic_Ys = mapnp([[5, 6, 7], [7, 6, 5]])
        self.assertRaises(ValueError, LMC,
                          basic_Xs, basic_Ys, kernels=[])

    def test_kernel_reconstruction(self):
        # TODO follow example in interpolation.py
        pass

    def test_normal_quadratic(self):
        # should agree with K_SKI (make a randomtest?)
        pass

    def test_optimization(self):
        # on a random example, NLL should decrease before/after.
        pass

    def test_1d_fit(self):
        # internally, fit should get within noise on training data
        # up to tolerance (use sin).
        pass

    def test_2d_fit_nocov(self):
        # internally, fit should get within noise on training data up
        # to tolerance (use sin/cos)
        # note coregionalization may be found even if not there - nonconvex
        # problem won't necessarily find the right solution -> don't check
        # params, just fit
        pass

    # TODO: test_2d_fit_cov - with covariance
    # TODO: test_coreg_nocov - requires rank 2, single kernel, l1 prior
    #       on coreg (should find identity matrix approx coregionalization
    #       after a couple random restarts (choose l2 err minimizing one,
    #       on data with no noise).
    # TODO: same as above, but find the covariance. (check we don't go to the
    #       identity)
