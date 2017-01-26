# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import unittest
import math

import numdifftools as nd
import numpy as np

from .matern32 import Matern32
from ..util.numpy_convenience import map_entries
from ..util.testing_utils import BasicModel, check_np_lists

class RBFTest(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.cases = np.arange(10.0)
        self.inv_lengthscale = 4
        self.testK = Matern32(self.inv_lengthscale, 'wierd-name')

    def f(self, r):
        scaled = math.sqrt(3) * self.inv_lengthscale * r
        return (1 + scaled) * math.exp(-scaled)

    def test_defaults(self):
        k = Matern32()
        self.assertEqual(k.inv_lengthscale, 1)
        self.assertEqual(k.name, 'matern32')

    def test_values(self):
        actual = self.testK.from_dist(self.cases)
        expected = map_entries(self.f, self.cases)
        np.testing.assert_almost_equal(actual, expected)

    def test_numerical_gradients(self):
        actual = self.testK.kernel_gradient(self.cases)
        def deriv(x):
            return nd.Derivative(lambda l: Matern32(l).from_dist(x))
        expected = [deriv(x)(self.inv_lengthscale)[0] for x in self.cases]
        check_np_lists(actual, [expected])

    def test_to_gpy(self):
        gpy = self.testK.to_gpy()
        self.assertEqual(gpy.name, self.testK.name)
        self.assertEqual(float(gpy.variance[0]), 1)
        self.assertEqual(gpy.input_dim, 1)
        self.assertAlmostEqual(gpy.lengthscale[0],
                               1 / self.inv_lengthscale)
        self.assertEqual(gpy.ARD, False)
        self.assertEqual(gpy.active_dims, [0])

    def test_optimization(self):
        dists = np.arange(10)
        Y = np.sin(np.arange(10))
        m = BasicModel(dists, Y, self.testK)
        ll_before = m.log_likelihood()
        m.optimize('lbfgsb')
        self.assertGreaterEqual(m.log_likelihood(), ll_before)

    def test_optimization_priors_one_step(self):
        # TODO: check that priors propogate
        pass
