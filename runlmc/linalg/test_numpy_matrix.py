# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import numpy as np
import scipy.linalg

from .test_matrix_base import MatrixTestBase
from .numpy_matrix import NumpyMatrix
from ..util.testing_utils import RandomTest

class NumpyMatrixTest(RandomTest, MatrixTestBase):

    def setUp(self):
        super().setUp()

        random = np.abs(np.hstack([np.random.rand(30), np.zeros(10)]))
        random[::-1].sort()
        random[0] += np.abs(random[1:]).sum()
        random_toep = scipy.linalg.toeplitz(random)

        up = lambda x: np.diag(np.arange(x) + 1)
        down = lambda x: up(x)[::-1, ::-1]

        self.eigtol = 1e-3
        self.examples = [NumpyMatrix(x) for x in [
            np.identity(1),
            np.identity(2),
            up(3),
            down(3),
            random_toep,
            self._rpsd(10),
            self._rpsd(100),
            np.kron(np.identity(2), np.identity(3) * self.eigtol / 2),
            np.kron(np.identity(2), np.identity(3) * self.eigtol),
            np.kron(np.identity(2), np.identity(3) * self.eigtol * 2)]]

    def test_as_numpy(self):
        for n in self.examples:
            np.testing.assert_array_equal(n.A, n.as_numpy())

    def test_shape_2d(self):
        A = np.arange(3 * 4 * 5).reshape(3, 4, 5)
        self.assertRaises(ValueError, NumpyMatrix, A)
        self.assertRaises(ValueError, NumpyMatrix, A.reshape(-1))

    def test_nonsymm(self):
        A = np.array([[1, 1], [0, 1]])
        self.assertRaises(ValueError, NumpyMatrix, A)
