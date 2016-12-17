# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import unittest

import numpy as np

from ..parameterization.model import Model
from .constant import Constant
from ..util.testing_utils import check_np_lists

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
        check_np_lists(grad, [np.ones(10)])

    def test_gradient_one_multi(self):
        z = Constant(1, 3, np.arange(3))
        grad = z.mean_gradient([np.arange(10), np.arange(2), np.arange(5)])
        check_np_lists(grad, [np.ones(10), np.ones(2), np.ones(5)])

    def test_optimization(self):
        # TODO: create a subclass dummy model that changes the params once
        # and test with sgd optimizer (set deriv to 0 after)
        # TODO: do same as above, but with a prior.
        Xs = [np.arange(10), np.arange(2), np.arange(5)]
        z = Constant(1, 3, np.arange(3))
