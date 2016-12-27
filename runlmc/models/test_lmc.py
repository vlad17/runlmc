# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import unittest

import numpy as np

from .lmc import LMC
from ..kern.rbf import RBF

class LMCTest(unittest.TestCase):

    def setUp(self):
        super().setUp()
        mapnp = lambda x: [np.array(i) for i in x]

        self.basic_kernels = [
            RBF(name='rbf1'),
            RBF(variance=2, inv_lengthscale=2, name='rbf2')]
        self.basic_Xs = mapnp([[0, 1, 2], [0.5, 1.5, 2.5]])
        self.basic_Ys = mapnp([[5, 6, 7], [7, 6, 5]])

    def test_no_kernel(self):
        self.assertRaises(ValueError, LMC,
                          self.basic_Xs, self.basic_Ys, kernels=[])
