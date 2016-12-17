# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import unittest

import numpy as np

from .mean_function import MeanFunction

class MeanFunctionTest(unittest.TestCase):

    def test_one_d_input(self):
        self.assertRaises(AssertionError, MeanFunction, 2, 1)
        self.assertRaises(AssertionError, MeanFunction, 2, 2)

    def test_output_dims_valid(self):
        MeanFunction(1, 1)
        MeanFunction(1, 10)

    def test_input_validation(self):
        mf = MeanFunction(1, 2)
        empty = np.array([])
        self.assertRaises(ValueError, mf._validate_inputs, [])
        self.assertRaises(
            ValueError, mf._validate_inputs, [empty, empty, empty])
        self.assertRaises(
            ValueError, mf._validate_inputs, [np.ones(1), np.ones((1, 1))])
        mf._validate_inputs([empty, empty])
        mf._validate_inputs([np.ones(10), np.ones(3)])
