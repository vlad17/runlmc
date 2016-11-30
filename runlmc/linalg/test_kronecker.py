# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

from functools import reduce
import unittest

import numpy as np
import scipy.linalg

from .test_matrix_base import DecomposableMatrixTestBase
from .kronecker import Kronecker
from .toeplitz import Toeplitz

class KroneckerTest(unittest.TestCase, DecomposableMatrixTestBase):

    def setUp(self):
        super().setUp()

        random = np.abs(np.hstack([np.random.rand(30), np.zeros(10)]))
        random[::-1].sort()
        random[0] += np.abs(random[1:]).sum()

        up = lambda x: np.diag(np.arange(x) + 1)
        down = lambda x: up(x)[::-1, ::-1]

        self.eigtol = 1e-3

        examples = [
            [up(1), down(1)],
            [up(3), down(2)],
            [up(2), down(3)],
            [scipy.linalg.hilbert(3), scipy.linalg.hilbert(3)],
            [self._rpsd(3), np.identity(2)],
            [self._rpsd(2), self._rpsd(3)],
            [up(3), Toeplitz(np.arange(10)[::-1] + 1)],
            [Toeplitz(random), self._rpsd(5)],
            [self._rpsd(100), self._rpsd(5)],
            [np.identity(2), np.identity(3) * self.eigtol / 2],
            [np.identity(2), np.identity(3) * self.eigtol],
            [np.identity(2), np.identity(3) * self.eigtol * 2]]

        self.examples = list(map(self._generate, examples))

    @staticmethod
    def _generate(mats):
        my_kron = reduce(Kronecker, mats)
        mats = [scipy.linalg.toeplitz(x.top) if isinstance(x, Toeplitz)
                else x for x in mats]
        np_kron = reduce(np.kron, mats)
        return my_kron, np_kron

    def test_empty(self):
        empty = np.array([[]])
        one = np.array([[1]])
        self.assertRaises(ValueError, Kronecker, empty, one)
        self.assertRaises(ValueError, Kronecker, one, empty)
        self.assertRaises(ValueError, Kronecker, empty, empty)

    def test_type(self):
        class Dummy:
            shape = (1, 1)
        self.assertRaises(TypeError, Kronecker, Dummy(), np.array([[1]]))
