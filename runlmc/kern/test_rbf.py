# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import unittest
import math

import numpy as np

from .rbf import RBF
from ..util.numpy_convenience import map_entries

class RBFTest(unittest.TestCase):

    def setUp(self):
        self.cases = np.arange(3.0)
        variance = 3
        self.rbf_f = lambda r: variance * math.exp(-0.5 * r * r)
        self.rbf_df = lambda r: -r * self.rbf_f(r)
        self.testK = RBF(variance, 'wierd-name')

    def test_defaults(self):
        k = RBF()
        self.assertEqual(k.variance, 1)
        self.assertEqual(k.name, 'rbf')

    def test_values(self):
        actual = self.testK.from_dist(self.cases)
        expected = map_entries(self.rbf_f, self.cases)
        np.testing.assert_almost_equal(actual, expected)

    def test_to_gpy(self):
        gpy = self.testK.to_gpy()
        self.assertEqual(gpy.name, self.testK.name)
        self.assertEqual(float(gpy.variance[0]), float(self.testK.variance[0]))
        self.assertEqual(gpy.input_dim, 1)
        self.assertEqual(gpy.lengthscale, [1])
        self.assertEqual(gpy.ARD, False)
        self.assertEqual(gpy.active_dims, [0])
