# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import unittest

import numpy as np

from .multigp import MultiGP
from ..mean.zero import Zero

class DummyMultiGP(MultiGP):
    def __init__(self, Xs, Ys, normalize, mean, name):
        super().__init__(Xs, Ys, mean_function=mean,
                         normalize=normalize, name=name)
        self._log_likelihood = 0
        self._n_param_changes = 0

    def _raw_predict(self, Xs):
        return [np.copy(X) for X in Xs], [np.fabs(X) for X in Xs]

    def log_likelihood(self):
        self._log_likelihood += 1
        return self._log_likelihood

    def parameters_changed(self):
        self._n_param_changes += 1

class MultiGPTest(unittest.TestCase):

    def test_empty(self):
        self.assertRaises(ValueError, DummyMultiGP,
                          [], [], False, None, '')

    def test_dim_X(self):
        Xs = [np.arange(2).reshape(1, -1)]
        Ys = [np.arange(2)]
        self.assertRaises(ValueError, DummyMultiGP,
                          Xs, Ys, False, None, '')

    def test_dim_Y(self):
        Xs = [np.arange(2)]
        Ys = [np.arange(2).reshape(1, -1)]
        self.assertRaises(ValueError, DummyMultiGP,
                          Xs, Ys, False, None, '')

    def test_lengths_unequal(self):
        Xs = [np.arange(2)]
        Ys = Xs + Xs
        self.assertRaises(ValueError, DummyMultiGP,
                          Xs, Ys, False, None, '')

    def test_mean_dims(self):
        Xs = [np.arange(2)]
        Ys = Xs
        mean = Zero(1, 1)
        mean.input_dim = 2 # hack to get around unimplemented multi-dim input
        self.assertRaises(ValueError, DummyMultiGP,
                          Xs, Ys, False, mean, '')
        mean = Zero(1, 2)
        self.assertRaises(ValueError, DummyMultiGP,
                          Xs, Ys, False, mean, '')

    def test_zero_sd(self):
        Xs = [np.arange(2)]
        Ys = [np.ones(2)]
        self.assertRaises(ValueError, DummyMultiGP,
                          Xs, Ys, False, None, '')


    def test_basic_creation(self):
        Xs = [np.arange(10)]
        Ys = Xs
        mean = Zero(1, 1)
        DummyMultiGP(Xs, Ys, True, mean, '')

    def test_multiout_creation(self):
        Xs = [np.arange(10), np.arange(10)]
        Ys = Xs
        mean = Zero(1, 2)
        DummyMultiGP(Xs, Ys, True, mean, '')

    def test_name(self):
        Xs = [np.arange(2)]
        Ys = Xs
        mean = Zero(1, 1)
        gp = DummyMultiGP(Xs, Ys, True, mean, 'hello')
        self.assertEqual(gp.name, 'hello')

    def test_normalization(self):
        Xs = [np.arange(4), np.arange(4)]
        Ys = [np.array([0, 2, 0, 2]), np.array([0, 3, 0, 3])]
        gp = DummyMultiGP(Xs, [np.copy(Y) for Y in Ys], True, None, '')

        self.assertEqual(np.std(gp.Ys[0]), 1)
        self.assertEqual(np.std(gp.Ys[1]), 1)
        self.assertEqual(np.mean(gp.Ys[0]), 0)
        self.assertEqual(np.mean(gp.Ys[1]), 0)

        sd1, sd2 = (np.std(Y) for Y in Ys)
        mu1, mu2 = (np.mean(Y) for Y in Ys)

        mu, var = gp.predict(Xs) # Dummy prediction returns Xs, abs(Xs)

        actual1, actual2 = mu
        np.testing.assert_allclose(Xs[0], (actual1 - mu1) / sd1)
        np.testing.assert_allclose(Xs[1], (actual2 - mu2) / sd2)

        var1, var2 = var
        np.testing.assert_allclose(Xs[0], var1 / sd1 / sd1)
        np.testing.assert_allclose(Xs[1], var2 / sd2 / sd2)

    def test_mean_function(self):
        pass

    # TODO: add tests for the following functionality
    #   - mean function should shift appropriately
    #   - derived class with its own priored param, mean, and _raw_predict):
    #   - Make sure dummy param, mean function perpetuate priors
    #   - Make sure dummy param, mean function are notified of gradient
    #   - Check above for kernel/gradient priors propogated as well
    #   - predict_quantiles works property for >2 tups
    # TODO(priors)
