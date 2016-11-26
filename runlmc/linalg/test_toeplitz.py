# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import unittest

import numpy as np
import scipy.linalg

from .toeplitz import SymmToeplitz

class ToeplitzTest(unittest.TestCase):

    def setUp(self):
        up = np.arange(100)
        down = np.copy(up[::-1])
        up = up / 2
        updown = np.add.accumulate(np.hstack([up, down]))[::-1]
        downup = np.hstack([down, up])

        random = np.abs(np.hstack([np.random.rand(30), np.zeros(10)]))
        random[::-1].sort()
        random[0] += np.abs(random[1:]).sum()
        rand_b = np.random.rand(40)

        self.examples = [(np.array(a), np.array(b)) for a, b in [
            ([1, 1], [2, 2]),
            ([1, -1], [-1, 1]),
            ([1, 3, 1], [2, 1, 1]),
            (np.arange(10)[::-1], np.arange(10)),
            (updown, downup),
            (random, rand_b)]]

    def test_bad_shape(self):
        two_d = np.arange(8).reshape(2, 4)
        self.assertRaises(ValueError, SymmToeplitz, two_d)
        empty = np.array([])
        self.assertRaises(ValueError, SymmToeplitz, empty)

    def test_bad_type(self):
        cplx = np.arange(5) * 1j
        self.assertRaises(Exception, SymmToeplitz, cplx)

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
            tol = 1e-15
            cg_solve = SymmToeplitz(top).solve(b, tol)
            np_solve = np.linalg.solve(toep, b)
            import math
            atol = tol * math.sqrt(np.linalg.cond(toep))
            np.testing.assert_allclose(cg_solve, np_solve, atol=atol, rtol=0)

    def test_eig(self):
        for top, _ in self.examples:
            top = self.examples[len(self.examples)-1][0]
            toep = scipy.linalg.toeplitz(top)
            np_vals = np.linalg.eigvalsh(toep)
            np_vals[::-1].sort()
            tol = 1e-4
            vals = SymmToeplitz(top).eig(tol)
            self.assertNotEqual(len(vals), 0)
            np.testing.assert_allclose(vals, np_vals[:len(vals)])
            self.assertTrue(all(vals > tol))
