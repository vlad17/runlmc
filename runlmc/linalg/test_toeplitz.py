# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import unittest

import numpy as np
import scipy.linalg

from .test_matrix_base import MatrixTestBase
from .toeplitz import Toeplitz

class ToeplitzTest(unittest.TestCase, MatrixTestBase):

    def setUp(self):
        super().setUp()

        random = np.abs(np.hstack([np.random.rand(30), np.zeros(10)]))
        random[::-1].sort()
        random[0] += np.abs(random[1:]).sum()

        up = lambda x: np.arange(x) + 1
        down = lambda x: up(x)[::-1]

        self.eigtol = 1e-3
        self.examples = [self._generate(x) for x in [
            [1],
            [1, 0],
            [1, 1],
            [0, 0],
            [1, -1],
            self._toep_eig(self.eigtol / 2, 5),
            self._toep_eig(self.eigtol, 5),
            self._toep_eig(self.eigtol * 2, 5),
            [1, 2, -1],
            up(10),
            down(10),
            random]]

    @staticmethod
    def _generate(x):
        x = np.array(x)
        msg = 'Toeplitz {}'.format(
            'size {}'.format(len(x)) if len(x) > 10 else x)
        return (Toeplitz(x), scipy.linalg.toeplitz(x), msg)

    def test_bad_shape(self):
        two_d = np.arange(8).reshape(2, 4)
        self.assertRaises(ValueError, Toeplitz, two_d)
        empty = np.array([])
        self.assertRaises(ValueError, Toeplitz, empty)

    def test_bad_type(self):
        cplx = np.arange(5) * 1j
        self.assertRaises(Exception, Toeplitz, cplx)
