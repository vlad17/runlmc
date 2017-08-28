# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import numpy as np
import scipy.linalg as la

from .test_matrix_base import MatrixTestBase
from .kronecker import Kronecker
from .numpy_matrix import NumpyMatrix
from .matrix import Matrix
from .sum_matrix import SumMatrix
from .toeplitz import Toeplitz
from ..util.testing_utils import RandomTest, exp_decr_toep


class SumMatrixTest(RandomTest, MatrixTestBase):

    def setUp(self):
        super().setUp()

        def up(x):
            return np.diag(np.arange(x) + 1)

        def down(x):
            return up(x)[::-1, ::-1]

        self.eigtol = 1e-3

        # Pathological Toeplitz matrices are dense in examples here
        # to test the sum_matrix algorithm; using the approximation
        # of eigenvalues in Toeplitz is too dangerous.
        examples = [
            # Square
            [up(1), up(1), up(1), np.diag(np.ones(1))],

            [up(3), down(3), up(3), np.diag(np.ones(3))],

            [self._rpsd(3), self._rpsd(3), np.diag(np.random.rand(3))],

            [la.toeplitz(self._toep_eig(1e-3 * i, 5))
             for i in range(1, 4)] + [np.diag(1e-3 * (1 + np.random.rand(6)))],

            [la.toeplitz(self._toep_eig(1e-6 * i, 5))
             for i in range(1, 4)] + [np.diag(1e-6 * (1 + np.random.rand(6)))],

            [la.toeplitz(self._toep_eig(1e-10 * i, 5))
             for i in range(1, 4)] +
            [np.diag(1e-10 * (1 + np.random.rand(6)))],

            [la.toeplitz(np.arange(10)[::-1] * i)
             for i in range(1, 4)] +
            [np.diag(1e-5 * (1 + np.random.rand(10)))],

            [Toeplitz(exp_decr_toep(5))
             for _ in range(5)] + [np.diag(np.ones(5) * 1e-4)],

            [Kronecker(NumpyMatrix(self._rpsd(2)), Toeplitz(exp_decr_toep(5)))
             for _ in range(5)] + [np.diag(np.ones(10) * 1e-4)],

            [Kronecker(NumpyMatrix(self._rpsd(2)),
                       NumpyMatrix(self._rpsd(2)))
             for _ in range(2)] + [np.diag(np.ones(4) * 1e-4)],

            [Toeplitz(exp_decr_toep(5)) for _ in range(5)] +

            [np.diag(np.ones(5) * 1e-4)],

            [Kronecker(NumpyMatrix(self._rpsd(2)),
                       Kronecker(NumpyMatrix(self._rpsd(2)),
                                 NumpyMatrix(self._rpsd(10))))
             for _ in range(2)] + [np.diag(np.ones(40) * 1e-4)],

            [self._rpsd(100) for _ in range(10)] +
            [np.diag(np.random.rand(100))],
            # Rectangle
            [np.random.rand(2, 3), np.random.rand(2, 3)],
            [np.random.rand(3, 2) for _ in range(4)],
            [np.random.rand(3, 4) for _ in range(4)]]

        self.examples = list(map(self._generate, examples))

    @staticmethod
    def _generate(mats):
        my_mats = [NumpyMatrix(x) if not isinstance(x, Matrix) else x
                   for x in mats]
        return SumMatrix(my_mats)

    def test_as_numpy(self):
        for s in self.examples:
            np_mat = sum(K.as_numpy() for K in s.Ks)
            np.testing.assert_allclose(s.as_numpy(), np_mat)

    def test_no_diff_shapes(self):
        sizes = [3, 3, 3, 4, 3]
        mats = [NumpyMatrix(np.identity(i)) for i in sizes]
        self.assertRaises(ValueError, SumMatrix, mats)

    def test_empty(self):
        self.assertRaises(ValueError, SumMatrix, [])
