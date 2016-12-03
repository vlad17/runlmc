# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import unittest

import numpy as np
import scipy.linalg

from .test_matrix_base import MatrixTestBase
from .numpy_matrix import NumpyMatrix
from .psd_matrix import PSDMatrix
from .sum_matrix import SumMatrix
from .toeplitz import Toeplitz

class SumMatrixTest(unittest.TestCase, MatrixTestBase):

    def setUp(self):
        super().setUp()

        up = lambda x: np.diag(np.arange(x) + 1)
        down = lambda x: up(x)[::-1, ::-1]

        self.eigtol = 1e-3

        examples = [
            ([up(1), up(1), up(1)], np.ones(1)),
            ([up(3), down(3), up(3)], np.ones(3)),
            ([self._rpsd(3), self._rpsd(3)], np.random.rand(3)),
            ([Toeplitz(self._toep_eig(1e-3 * i, 5)) for i in range(1, 3)],
             1e-3 * (1 + np.random.rand(6))),
            ([Toeplitz(self._toep_eig(1e-6 * i, 5)) for i in range(1, 3)],
             1e-6 * (1 + np.random.rand(6))),
            ([Toeplitz(self._toep_eig(1e-10 * i, 5)) for i in range(1, 3)],
             1e-10 * (1 + np.random.rand(6))),
            ([Toeplitz(np.arange(10)[::-1] * i) for i in range(1, 3)],
             1e-5 * (1 + np.random.rand(10))),
            ([self._rpsd(100) for _ in range(10)], np.random.rand(100))]

        self.examples = list(map(self._generate, examples))

    @staticmethod
    def _generate(mats_and_noise):
        mats, noise = mats_and_noise
        my_mats = [NumpyMatrix(x) if not isinstance(x, PSDMatrix) else x
                   for x in mats]
        as_np = (scipy.linalg.toeplitz(x.top) if isinstance(x, Toeplitz) else x
                 for x in mats)
        np_mat = sum(as_np) + np.diag(noise)
        return SumMatrix(my_mats, noise), np_mat

    def test_empty(self):
        sum_mat = SumMatrix([], np.ones(3))
        np.testing.assert_allclose(sum_mat.matvec(np.arange(3)), np.arange(3))
        self.assertRaises(ValueError, SumMatrix, [], np.array([]))

    def test_no_diff_shapes(self):
        sizes = [3, 3, 3, 4, 3]
        mats = [NumpyMatrix(np.identity(i)) for i in sizes]
        self.assertRaises(ValueError, SumMatrix, mats, np.identity(3))

    def test_neg_noise(self):
        self.assertRaises(ValueError,
                          SumMatrix,
                          [np.identity(2)],
                          -np.ones(2))

    def test_noise_mishapen(self):
        self.assertRaises(ValueError,
                          SumMatrix,
                          [np.identity(2)],
                          np.ones(3))

    def test_logdet(self):
        for my_mat, np_mat in self.examples:
            sign, logdet = np.linalg.slogdet(np_mat)
            self.assertGreater(sign, 0)
            my_logdet = my_mat.logdet()
            rel_err = abs(my_logdet - logdet) / abs(logdet)
            self.assertGreaterEqual(my_logdet, logdet,
                                    msg='\n{!s}\n'.format(my_mat))
            self.assertGreaterEqual(2, rel_err,
                                    msg='\n{!s}\n'.format(my_mat))
