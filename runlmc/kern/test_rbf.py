# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import unittest
import math

import numpy as np
import scipy.linalg

from .rbf import RBF
from ..parameterization.model import Model
from ..util.numpy_convenience import map_entries
from ..util.testing_utils import check_np_lists, SingleGradOptimizer

class BasicModel(Model):
    def __init__(self, dists, Y, kern):
        super().__init__('single-kern')
        self.link_parameter(kern)
        self.dists = dists
        self.kern = kern
        self.Y = Y

    def log_likelihood(self):
        K_top = self.kern.from_dist(self.dists)
        KinvY = scipy.linalg.solve_toeplitz(K_top, self.Y)
        # Prevent slight negative eigenvalues from roundoff.
        sign, logdet = np.linalg.slogdet(
            scipy.linalg.toeplitz(K_top) + 1e-10 * np.identity(len(K_top)))
        assert sign > 0, sign
        return -0.5 * self.Y.dot(KinvY) - 0.5 * logdet

    def parameters_changed(self):
        # maximize -0.5 * (y . K^-1 y) - 0.5 log |K|
        # gradient wrt t is 0.5 tr((a a^T - K^-1)dK/dt), a = K^-1 a
        K_top = self.kern.from_dist(self.dists)
        a = scipy.linalg.solve_toeplitz(K_top, self.Y)
        all_grad = self.kern.kernel_gradient(self.dists)
        likelihood_grad = np.zeros(len(all_grad))
        for i, grad in enumerate(all_grad):
            dKdt = scipy.linalg.toeplitz(grad)
            Kinv_dKdt = scipy.linalg.solve_toeplitz(K_top, dKdt)
            aaT_dKdt = np.outer(a, dKdt.dot(a))
            trace = np.trace(aaT_dKdt - Kinv_dKdt)
            likelihood_grad[i] = 0.5 * trace

        self.kern.update_gradient(likelihood_grad)


class RBFTest(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.cases = np.arange(10.0)
        self.variance = 3
        self.inv_lengthscale = 4
        self.rbf_f = lambda r: self.variance * math.exp(-0.5 * r * r)
        self.rbf_df = lambda r: -r * self.rbf_f(r)
        self.testK = RBF(self.variance, self.inv_lengthscale, 'wierd-name')

    def f(self, r):
        exp = math.exp(-0.5 * self.inv_lengthscale * r * r)
        return self.variance * exp

    def df_dv(self, r):
        exp = math.exp(-0.5 * self.inv_lengthscale * r * r)
        return exp

    def df_dl(self, r):
        exp = math.exp(-0.5 * self.inv_lengthscale * r * r)
        return exp * -0.5 * r * r * self.variance

    def test_defaults(self):
        k = RBF()
        self.assertEqual(k.variance, 1)
        self.assertEqual(k.inv_lengthscale, 1)
        self.assertEqual(k.name, 'rbf')

    def test_values(self):
        actual = self.testK.from_dist(self.cases)
        expected = map_entries(self.f, self.cases)
        np.testing.assert_almost_equal(actual, expected)

    def test_gradients(self):
        actual = self.testK.kernel_gradient(self.cases)
        expected = [
            map_entries(self.df_dv, self.cases),
            map_entries(self.df_dl, self.cases)]
        check_np_lists(actual, expected)

    def test_to_gpy(self):
        gpy = self.testK.to_gpy()
        self.assertEqual(gpy.name, self.testK.name)
        self.assertEqual(float(gpy.variance[0]), float(self.testK.variance[0]))
        self.assertEqual(gpy.input_dim, 1)
        self.assertAlmostEqual(gpy.lengthscale[0],
                               self.inv_lengthscale ** -0.5)
        self.assertEqual(gpy.ARD, False)
        self.assertEqual(gpy.active_dims, [0])

    def test_optimization_one_step(self):
        dists = np.arange(10)
        opt = SingleGradOptimizer()
        Y = np.sin(np.arange(10))
        m = BasicModel(dists, Y, self.testK)
        ll_before = m.log_likelihood()
        m.optimize(opt)
        ll_after = m.log_likelihood()
        self.assertGreater(ll_after, ll_before)
        self.assertNotEqual(self.variance, float(self.testK.variance[0]))
        self.assertNotEqual(self.inv_lengthscale,
                            float(self.testK.inv_lengthscale[0]))

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
