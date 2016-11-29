# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import unittest

import numpy as np

from .psd_matrix import PSDMatrix

class PSDMatrixTest(unittest.TestCase):

    def test_attributes(self):
        m = PSDMatrix(3)
        self.assertEqual(m.dtype, np.float64)
        self.assertEqual(m.shape, (3, 3))

    def test_linear_operator(self):
        m = PSDMatrix(3)
        lo = m.as_linear_operator()
        self.assertEqual(lo.dtype, np.float64)
        self.assertEqual(lo.shape, (3, 3))

    def test_bad_size(self):
        self.assertRaises(ValueError, PSDMatrix, 0)
        self.assertRaises(ValueError, PSDMatrix, -2)
