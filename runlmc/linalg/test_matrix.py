# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import unittest

import numpy as np

from .matrix import Matrix

class MatrixTest(unittest.TestCase):

    def test_attributes(self):
        m = Matrix(3)
        self.assertEqual(m.dtype, np.float64)
        self.assertEqual(m.shape, (3, 3))

    def test_bad_size(self):
        self.assertRaises(ValueError, Matrix, 0)
        self.assertRaises(ValueError, Matrix, -2)
