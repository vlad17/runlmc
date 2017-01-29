# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import unittest

import numpy as np

from .symmetric_matrix import SymmetricMatrix

class SymmetricMatrixTest(unittest.TestCase):

    def test_attributes(self):
        m = SymmetricMatrix(3)
        self.assertEqual(m.dtype, np.float64)
        self.assertEqual(m.shape, (3, 3))

    def test_linear_operator(self):
        m = SymmetricMatrix(3)
        lo = m.as_linear_operator()
        self.assertEqual(lo.dtype, np.float64)
        self.assertEqual(lo.shape, (3, 3))

    def test_bad_size(self):
        self.assertRaises(ValueError, SymmetricMatrix, 0)
        self.assertRaises(ValueError, SymmetricMatrix, -2)
