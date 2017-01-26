# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import unittest
import math

import numdifftools as nd
import numpy as np

from .std_periodic import StdPeriodic
from ..util.numpy_convenience import map_entries
from ..util.testing_utils import BasicModel, check_np_lists

class StdPeriodicTest(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.cases = np.arange(10.0)
        self.inv_lengthscale = 4
        self.period = 7
        self.testK = StdPeriodic(
            self.inv_lengthscale, self.period, 'wierd-name')

    def f(self, r):
        exp = self.inv_lengthscale * np.sin(np.pi * r / self.period) ** 2
        return math.exp(exp * -0.5)

    def df_dl(self, r):
        exp_nolen = -0.5 * np.sin(np.pi * r / self.period) ** 2
        return exp_nolen * math.exp(exp_nolen * self.inv_lengthscale)

    def df_dT(self, r):
        sin = (
            -0.5 * self.inv_lengthscale *
            np.sin(np.pi * r / self.period) ** 2)
        dsin = -1 * self.inv_lengthscale * np.sin(np.pi * r / self.period)
        dsin *= np.cos(np.pi * r / self.period)
        dsin *= -1 * np.pi * r / self.period ** 2
        return math.exp(sin) * dsin

    def test_defaults(self):
        k = StdPeriodic()
        self.assertEqual(k.inv_lengthscale, 1)
        self.assertEqual(k.period, 1)
        self.assertEqual(k.name, 'std_periodic')

    def test_values(self):
        actual = self.testK.from_dist(self.cases)
        expected = map_entries(self.f, self.cases)
        np.testing.assert_almost_equal(actual, expected)

    def test_gradients(self):
        actual = self.testK.kernel_gradient(self.cases)
        expected = [
            map_entries(self.df_dl, self.cases),
            map_entries(self.df_dT, self.cases)]
        check_np_lists(actual, expected)

    def test_numerical_gradients(self):
        actual = self.testK.kernel_gradient(self.cases)
        def deriv(x):
            return nd.Gradient(lambda lT:
                               StdPeriodic(lT[0], lT[1]).from_dist(x))
        expected = np.array([deriv(x)([self.inv_lengthscale, self.period])
                             for x in self.cases]).T
        check_np_lists(actual, expected, atol=1e-5)

    def test_to_gpy(self):
        gpy = self.testK.to_gpy()
        self.assertEqual(gpy.name, self.testK.name)
        self.assertEqual(float(gpy.variance[0]), 1)
        self.assertEqual(float(gpy.period[0]), self.period)
        self.assertEqual(gpy.input_dim, 1)
        self.assertAlmostEqual(gpy.lengthscale[0],
                               self.inv_lengthscale ** -0.5)
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
