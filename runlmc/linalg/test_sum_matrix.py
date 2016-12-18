# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import numpy as np
import scipy.linalg

from .test_matrix_base import MatrixTestBase
from .kronecker import Kronecker
from .numpy_matrix import NumpyMatrix
from .psd_matrix import PSDMatrix
from .sum_matrix import SumMatrix
from .toeplitz import Toeplitz
from ..util.testing_utils import RandomTest, exp_decr_toep

class SumMatrixTest(RandomTest, MatrixTestBase):

    def setUp(self):
        super().setUp()

        up = lambda x: np.diag(np.arange(x) + 1)
        down = lambda x: up(x)[::-1, ::-1]

        self.eigtol = 1e-3

        # Pathological Toeplitz matrices are dense in examples here
        # to test the sum_matrix algorithm; using the approximation
        # of eigenvalues in Toeplitz is too dangerous.
        examples = [
            ([up(1), up(1), up(1)], np.ones(1)),
            ([up(3), down(3), up(3)], np.ones(3)),
            ([self._rpsd(3), self._rpsd(3)], np.random.rand(3)),
            ([scipy.linalg.toeplitz(self._toep_eig(1e-3 * i, 5))
              for i in range(1, 4)],
             1e-3 * (1 + np.random.rand(6))),
            ([scipy.linalg.toeplitz(self._toep_eig(1e-6 * i, 5))
              for i in range(1, 4)],
             1e-6 * (1 + np.random.rand(6))),
            ([scipy.linalg.toeplitz(self._toep_eig(1e-10 * i, 5))
              for i in range(1, 4)],
             1e-10 * (1 + np.random.rand(6))),
            ([scipy.linalg.toeplitz(np.arange(10)[::-1] * i)
              for i in range(1, 4)],
             1e-5 * (1 + np.random.rand(10))),
            ([Toeplitz(exp_decr_toep(5)) for _ in range(5)],
             np.ones(5) * 1e-4),
            ([Kronecker(self._rpsd(2), Toeplitz(exp_decr_toep(5)))
              for _ in range(5)],
             np.ones(10) * 1e-4),
            ([Kronecker(self._rpsd(2), self._rpsd(2))
              for _ in range(2)],
             np.ones(4) * 1e-4),
            ([Toeplitz(exp_decr_toep(5)) for _ in range(5)],
             np.ones(5) * 1e-4),
            ([Kronecker(self._rpsd(2),
                        Kronecker(self._rpsd(2), self._rpsd(10)))
              for _ in range(2)],
            ([self._rpsd(100) for _ in range(10)], np.random.rand(100))]

        self.examples = list(map(self._generate, examples))

    @staticmethod
    def _generate(mats_and_noise):
        mats, noise = mats_and_noise
        my_mats = [NumpyMatrix(x) if not isinstance(x, PSDMatrix) else x
                   for x in mats]
        return SumMatrix(my_mats, noise)

    def test_as_numpy(self):
        for s in self.examples:
            np_mat = sum(K.as_numpy() for K in s.Ks) + np.diag(s.noise)
            np.testing.assert_allclose(s.as_numpy(), np_mat)

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
        for my_mat in self.examples:
            np_mat = my_mat.as_numpy()
            sign, logdet = np.linalg.slogdet(np_mat)
            self.assertGreater(sign, 0)
            my_logdet = my_mat.logdet()
            msg = '\nmy logdet {} np logdet {}\n{!s}\n'.format(
                my_logdet, logdet, my_mat)

            # If we're very close to being singular then the bound isn't
            # so great; just make sure we have an upper bound
            if logdet < 0:
                self.assertGreaterEqual(my_logdet, logdet, msg=msg)
            else:
                rel_err = abs(logdet - my_logdet)
                rel_err /= 1 if logdet == 0 else abs(logdet)
                self.assertGreaterEqual(0.5, rel_err, msg=msg)
