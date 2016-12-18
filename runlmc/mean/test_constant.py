# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import unittest

import numpy as np

from ..parameterization.model import Model
from .constant import Constant
from ..util.testing_utils import check_np_lists, SingleGradOptimizer

class BasicModel(Model):
    def __init__(self, Xs, mean):
        super().__init__('single-mean')
        self.link_parameter(mean)
        self.Xs = Xs
        self.mean = mean

    def log_likelihood(self):
        return -sum(
            np.square(X - f).sum() / 2
            for X, f in zip(self.Xs, self.mean.f(self.Xs)))

    def parameters_changed(self):
        all_grad = self.mean.mean_gradient(self.Xs)
        fs = self.mean.f(self.Xs)
        likelihood_grad = np.zeros(len(all_grad))
        # maximize -|X-f|^2/2 (converge to mean of each output with no prior)
        # derivative of above is -df . (f - X)
        for i, grads in enumerate(all_grad):
            for X, f, grad in zip(self.Xs, fs, grads):
                likelihood_grad[i] -= grad.dot(f - X)

        self.mean.update_gradient(likelihood_grad)

class ConstantTest(unittest.TestCase):

    def test_output_init_mismatch(self):
        self.assertRaises(ValueError, Constant, 1, 3, np.zeros(2))
        self.assertRaises(ValueError, Constant, 1, 3, np.zeros(4))

    def test_default(self):
        z = Constant(1, 3)
        out = z.f([np.arange(10), np.arange(2), np.arange(5)])
        check_np_lists(out, [np.zeros(10), np.zeros(2), np.zeros(5)])

    def test_f_one_one(self):
        z = Constant(1, 1, np.ones(1))
        out = z.f([np.arange(10)])
        check_np_lists(out, [np.ones(10)])

    def test_f_one_multi(self):
        z = Constant(1, 3, np.arange(3))
        out = z.f([np.arange(10), np.arange(2), np.arange(5)])
        check_np_lists(out, [np.zeros(10), np.ones(2), 2 * np.ones(5)])

    def test_gradient_one_one(self):
        z = Constant(1, 1, np.ones(1))
        grad = z.mean_gradient([np.arange(10)])
        check_np_lists(grad, [[np.ones(10)]])

    def test_gradient_one_multi(self):
        z = Constant(1, 3, np.arange(3))
        grad = z.mean_gradient([np.arange(10), np.arange(2), np.arange(5)])
        first, second, third = (
            [np.zeros(10), np.zeros(2), np.zeros(5)] for _ in range(3))
        first[0] += 1
        second[1] += 1
        third[2] += 1
        for actual, expected in zip(grad, [first, second, third]):
            check_np_lists(actual, expected)

    def test_optimization_one_step(self):
        Xs = [np.arange(10), np.arange(2), -np.arange(5)]
        z = Constant(1, 3, np.ones(3))
        opt = SingleGradOptimizer()
        m = BasicModel(Xs, z)
        m.optimize(opt)
        for grad, X in zip(opt.gradient_observed, Xs):
            self.assertEqual(grad, X.sum() - len(X))

    def test_optimization(self):
        Xs = [np.arange(10), np.arange(2), -np.arange(5)]
        z = Constant(1, 3, np.ones(3))
        m = BasicModel(Xs, z)
        m.optimize('lbfgsb')
        out = z.f(Xs)
        means = [np.repeat(X.mean(), len(X)) for X in Xs]
        check_np_lists(out, means, atol=1e-5)

    def test_optimization_priors_one_step(self):
        # TODO: check that priors propogate
        pass
