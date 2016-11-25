# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import unittest

import numpy as np
import scipy.linalg
import scipy.sparse.linalg

from .toeplitz import SymmToeplitz

class ToeplitzTest(unittest.TestCase):

    def setUp(self):
        self.examples = [(np.array(a), np.array(b)) for a, b in [
            ([1, 1], [2, 2]),
            ([1, -1], [-1, 1]),
            ([1, 3, 1], [2, 1, 1]),
            (np.arange(10), np.arange(10)[::-1])]]

    def test_bad_shape(self):
        two_d = np.arange(8).reshape(2, 4)
        self.assertRaises(ValueError, SymmToeplitz, two_d)
        empty = np.array([])
        self.assertRaises(ValueError, SymmToeplitz, empty)

    def test_matvec(self):
        for top, x in self.examples:
            my_dot = SymmToeplitz(top).matvec(x)
            toep = scipy.linalg.toeplitz(top)
            numpy_dot = toep.dot(x)
            np.testing.assert_allclose(my_dot, numpy_dot)

    def test_cg(self):
        for top, b in self.examples:
            toep = scipy.linalg.toeplitz(top)
            if np.linalg.matrix_rank(toep) < len(top):
                continue
            A = scipy.sparse.linalg.aslinearoperator(SymmToeplitz(top))
            tol = 1e-6
            cg_solve, success = scipy.sparse.linalg.cg(A, b, tol=tol)
            self.assertEqual(success, 0)
            np_solve = np.linalg.solve(toep, b)
            np.testing.assert_allclose(cg_solve, np_solve, atol=(tol * 2))
