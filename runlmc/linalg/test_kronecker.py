# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

from functools import reduce

import numpy as np
import scipy.linalg as la

from .matrix import Matrix
from .kronecker import Kronecker
from .test_matrix_base import MatrixTestBase
from .toeplitz import Toeplitz
from .numpy_matrix import NumpyMatrix
from ..util import testing_utils as utils


class KroneckerTest(utils.RandomTest, MatrixTestBase):

    def setUp(self):
        super().setUp()

        random = np.abs(np.hstack([np.random.rand(30), np.zeros(10)]))
        random[::-1].sort()
        random[0] += np.abs(random[1:]).sum()

        def up(x):
            return np.diag(np.arange(x) + 1)

        def down(x):
            return up(x)[::-1, ::-1]

        self.eigtol = 1e-3

        self.raw_examples = [
            # Square
            [up(1), down(1)],
            [up(3), down(2)],
            [up(2), down(3)],
            [la.hilbert(3), la.hilbert(3)],
            [self._rpsd(3), np.identity(2)],
            [self._rpsd(2), self._rpsd(3)],
            [up(3), Toeplitz(np.arange(10)[::-1] + 1)],
            [Toeplitz(random), self._rpsd(5)],
            [self._rpsd(100), self._rpsd(5)],
            [np.identity(2), np.identity(3) * self.eigtol / 2],
            [np.identity(2), np.identity(3) * self.eigtol],
            [np.identity(2), np.identity(3) * self.eigtol * 2],
            [self._rpsd(5), Toeplitz(utils.exp_decr_toep(10))],
            [self._rpsd(5), Toeplitz(utils.exp_decr_toep(100))],
            [Toeplitz(utils.exp_decr_toep(10)),
             Toeplitz(utils.exp_decr_toep(10))],
            [np.random.rand(2, 2) for _ in range(4)],
            [up(2), down(2), up(2)],
            # Rectangle
            [np.random.rand(2, 3), up(1)],
            [np.random.rand(2, 3), np.random.rand(3, 2)],
            [np.random.rand(4, 3), np.random.rand(5, 2), np.random.rand(1, 2)]]
        self.raw_examples = [[x if isinstance(x, Matrix) else NumpyMatrix(x)
                              for x in ls] for ls in self.raw_examples]

        self.examples = [reduce(Kronecker, x) for x in self.raw_examples]

    def test_shape(self):
        for k, raw in zip(self.examples, self.raw_examples):
            as_np = [x.as_numpy() for x in raw]
            self.assertEqual(k.shape, reduce(np.kron, as_np).shape)

    def test_as_numpy(self):
        for k, raw in zip(self.examples, self.raw_examples):
            as_np = [x.as_numpy() for x in raw]
            np.testing.assert_array_equal(
                k.as_numpy(), reduce(np.kron, as_np))
