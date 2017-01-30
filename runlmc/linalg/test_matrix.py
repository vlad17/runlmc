# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import unittest

import numpy as np

from .matrix import Matrix

class MatrixTest(unittest.TestCase):

    def test_attributes(self):
        m = Matrix(3, 5)
        self.assertEqual(m.dtype, np.float64)
        self.assertEqual(m.shape, (3, 5))

    def test_linear_operator(self):
        m = Matrix(3, 5)
        lo = m.as_linear_operator()
        self.assertEqual(lo.dtype, np.float64)
        self.assertEqual(lo.shape, (3, 5))

    def test_is_square(self):
        m = Matrix(3, 2)
        self.assertFalse(m.is_square())
        m = Matrix(3, 3)
        self.assertTrue(m.is_square())

    def test_bad_size(self):
        self.assertRaises(ValueError, Matrix, 0, 3)
        self.assertRaises(ValueError, Matrix, 3, -2)
        self.assertRaises(ValueError, Matrix, 0, 0)
