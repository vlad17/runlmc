# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import unittest

import numpy as np

from ..parameterization.model import Model
from ..util.testing_utils import check_np_lists
from .zero import Zero

class ZeroTest(unittest.TestCase):

    def test_f_one_one(self):
        z = Zero(1, 1)
        out = z.f([np.arange(10)])
        check_np_lists(out, [np.zeros(10)])

    def test_f_one_multi(self):
        z = Zero(1, 3)
        out = z.f([np.arange(10), np.arange(2), np.arange(5)])
        check_np_lists(out, [np.zeros(10), np.zeros(2), np.zeros(5)])

    def test_gradient_one_one(self):
        z = Zero(1, 1)
        grad = z.mean_gradient([np.arange(10)])
        check_np_lists(grad, [])

    def test_gradient_one_multi(self):
        z = Zero(1, 3)
        grad = z.mean_gradient([np.arange(10), np.arange(2), np.arange(5)])
        check_np_lists(grad, [])
