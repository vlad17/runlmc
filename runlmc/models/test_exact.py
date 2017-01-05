# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import unittest
import warnings

import numpy as np
from GPy.models import GPCoregionalizedRegression
from GPy.util.multioutput import LCM

from .exact import ExactLMC
from ..kern.rbf import RBF

class ExactLMCTest(unittest.TestCase):

    def setUp(self):
        super().setUp()
        mapnp = lambda x: [np.array(i) for i in x]

        self.basic_kernels = [
            RBF(name='rbf1'),
            RBF(inv_lengthscale=2, name='rbf2')]
        self.basic_Xs = mapnp([[0, 1, 2], [0.5, 1.5, 2.5]])
        self.basic_Ys = mapnp([[5, 6, 7], [7, 6, 5]])
        self.basic_predict_Xs = mapnp([[0.5, 1.5], [1.0, 2.0]])
        self.basic_meta = [0, 0, 1, 1]
        basic_gpy_ks = [k.to_gpy() for k in self.basic_kernels]
        self.basic_ranks = [1, 1]
        self.basic_gpy_K = LCM(
            input_dim=1,
            num_outputs=2,
            kernels_list=basic_gpy_ks,
            W_rank=1)

    def generate_basic(self):
        basic = ExactLMC(
            self.basic_Xs,
            self.basic_Ys,
            self.basic_kernels,
            self.basic_ranks,
            name='basic')
        basic_gpy = GPCoregionalizedRegression(
            [X.reshape(-1, 1) for X in self.basic_Xs],
            [Y.reshape(-1, 1) for Y in self.basic_Ys],
            self.basic_gpy_K)

        # Need to make sure we start from the same coefficients.
        # This may be a bit more involved when we stop using GPy for
        # the exact solution.
        basic_gpy.sum.ICM0[:] = basic.gpy_model.LCM.ICM0[:]
        basic_gpy.sum.ICM1[:] = basic.gpy_model.LCM.ICM1[:]

        return basic, basic_gpy

    def evaluate(self, runlmc, gpy, Xs, meta):
        mu, var = runlmc.predict(Xs)
        mu = np.vstack(mu).reshape(-1)
        var = np.vstack(var).reshape(-1)
        quantiles = runlmc.predict_quantiles(Xs)
        quantiles = np.vstack(quantiles).reshape(-1, 2)

        meta = np.array(meta).reshape(-1, 1)
        Xs = np.hstack([np.hstack(Xs).reshape(-1, 1), meta])
        mu_gpy, var_gpy = (x.reshape(-1) for x in gpy.predict(
            Xs,
            Y_metadata={'output_index': meta}))
        quantiles_gpy = np.hstack(gpy.predict_quantiles(
            Xs,
            (2.5, 97.5),
            Y_metadata={'output_index': meta}))

        np.testing.assert_almost_equal(mu, mu_gpy)
        np.testing.assert_almost_equal(var, var_gpy)
        np.testing.assert_almost_equal(quantiles, quantiles_gpy)

    def test_no_normalizer(self):
        basic, _ = self.generate_basic()
        self.assertEqual(basic.normalizer, None)

    def test_predict_format(self):
        basic, _ = self.generate_basic()
        Ys = basic.predict(self.basic_predict_Xs)
        shape_Xs = list(map(len, self.basic_predict_Xs))
        shape_Ys = list(map(len, Ys))
        self.assertEqual(shape_Xs, shape_Ys)

    def test_gpy_basic_no_opt(self):
        basic, basic_gpy = self.generate_basic()
        self.evaluate(basic, basic_gpy, self.basic_predict_Xs, self.basic_meta)

    def test_gpy_basic_full_opt(self):
        basic, basic_gpy = self.generate_basic()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            basic.optimize()
            basic_gpy.optimize()
            self.evaluate(basic, basic_gpy, self.basic_predict_Xs,
                          self.basic_meta)
