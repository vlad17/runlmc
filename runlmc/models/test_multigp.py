# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import unittest

import numpy as np
import scipy.stats

from .multigp import MultiGP
from ..util.testing_utils import check_np_lists

class DummyMultiGP(MultiGP):
    def __init__(self, Xs, Ys, normalize, name):
        super().__init__(Xs, Ys, normalize=normalize, name=name)
        self._log_likelihood = 0
        self._n_param_changes = 0

    def _raw_predict(self, Xs):
        return [np.copy(X).astype(float) for X in Xs], [np.fabs(X) for X in Xs]

    def log_likelihood(self):
        self._log_likelihood += 1
        return self._log_likelihood

    def parameters_changed(self):
        self._n_param_changes += 1

class MultiGPTest(unittest.TestCase):

    def test_empty(self):
        self.assertRaises(ValueError, DummyMultiGP,
                          [], [], False, '')

    def test_dim_X(self):
        Xs = [np.arange(2).reshape(1, -1)]
        Ys = [np.arange(2)]
        self.assertRaises(ValueError, DummyMultiGP,
                          Xs, Ys, False, '')

    def test_dim_Y(self):
        Xs = [np.arange(2)]
        Ys = [np.arange(2).reshape(1, -1)]
        self.assertRaises(ValueError, DummyMultiGP,
                          Xs, Ys, False, '')

    def test_lengths_unequal(self):
        Xs = [np.arange(2)]
        Ys = Xs + Xs
        self.assertRaises(ValueError, DummyMultiGP,
                          Xs, Ys, False, '')

    def test_zero_sd(self):
        Xs = [np.arange(2)]
        Ys = [np.ones(2)]
        self.assertRaises(ValueError, DummyMultiGP,
                          Xs, Ys, False, '')

    def test_basic_creation(self):
        Xs = [np.arange(10)]
        Ys = Xs
        DummyMultiGP(Xs, Ys, True, '')

    def test_multiout_creation(self):
        Xs = [np.arange(10), np.arange(10)]
        Ys = Xs
        DummyMultiGP(Xs, Ys, True, '')

    def test_name(self):
        Xs = [np.arange(2)]
        Ys = Xs
        gp = DummyMultiGP(Xs, Ys, True, 'hello')
        self.assertEqual(gp.name, 'hello')

    def test_predict(self):
        Xs = [np.arange(4), np.arange(4)]
        Ys = [np.array([0, 2, 0, 2]), np.array([0, 3, 0, 3])]
        gp = DummyMultiGP(Xs, [np.copy(Y) for Y in Ys], False, '')

        sd1, sd2 = (np.std(Y) for Y in Ys)
        mu1, mu2 = (np.mean(Y) for Y in Ys)
        self.assertEqual(np.std(gp.Ys[0]), sd1)
        self.assertEqual(np.std(gp.Ys[1]), sd2)
        self.assertEqual(np.mean(gp.Ys[0]), mu1)
        self.assertEqual(np.mean(gp.Ys[1]), mu2)

        mu, var = gp.predict(Xs)
        check_np_lists(Xs, mu)
        check_np_lists(Xs, var) # Xs already positive

    def test_normalization(self):
        Xs = [np.arange(4), np.arange(4)]
        Ys = [np.array([0, 2, 0, 2]), np.array([0, 3, 0, 3])]
        gp = DummyMultiGP(Xs, [np.copy(Y) for Y in Ys], True, '')

        self.assertEqual(np.std(gp.Ys[0]), 1)
        self.assertEqual(np.std(gp.Ys[1]), 1)
        self.assertEqual(np.mean(gp.Ys[0]), 0)
        self.assertEqual(np.mean(gp.Ys[1]), 0)

        sds = [np.std(Y) for Y in Ys]
        means = [np.mean(Y) for Y in Ys]

        mu, var = gp.predict(Xs)
        mu = [(actual - mean) / sd for actual, mean, sd in zip(mu, means, sds)]
        var = [actual / sd / sd for actual, sd in zip(var, sds)]

        check_np_lists(Xs, mu)
        check_np_lists(Xs, var)

    @staticmethod
    def create_quantile(mean, variance, quantiles):
        return mean + scipy.stats.norm.ppf(quantiles) * np.sqrt(variance)

    def test_predict_quantiles(self):
        Xs = [np.arange(4), np.arange(4)]
        Ys = [np.array([0, 2, 0, 2]), np.array([0, 3, 0, 3])]
        gp = DummyMultiGP(Xs, [np.copy(Y) for Y in Ys], False, '')

        quantiles = np.array([1, 10, 50, 99])
        actual = gp.predict_quantiles(Xs, quantiles)

        expected = [
            np.vstack([self.create_quantile(mean, var, quantiles / 100)
                       for mean, var in zip(X, np.fabs(X))])
            for X in Xs]

        check_np_lists(actual, expected)

    def test_predict_quantiles_with_norm(self):
        Xs = [np.arange(4), np.arange(4)]
        Ys = [np.array([0, 2, 0, 2]), np.array([0, 3, 0, 3])]
        gp = DummyMultiGP(Xs, [np.copy(Y) for Y in Ys], True, '')

        sds = [np.std(Y) for Y in Ys]
        means = [np.mean(Y) for Y in Ys]

        quantiles = np.array([1, 10, 50, 99])
        actual = gp.predict_quantiles(Xs, quantiles)

        expected = [
            np.vstack([self.create_quantile(
                mean + sd * unnorm_mean, sd * sd * unnorm_var, quantiles / 100)
                       for unnorm_mean, unnorm_var in zip(X, np.fabs(X))])
            for mean, sd, X in zip(means, sds, Xs)]

        check_np_lists(actual, expected)
