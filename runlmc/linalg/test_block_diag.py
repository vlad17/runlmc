# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import numpy as np
import scipy.linalg as la

from .test_matrix_base import MatrixTestBase
from .matrix import Matrix
from .numpy_matrix import NumpyMatrix
from .block_diag import BlockDiag
from ..util.testing_utils import RandomTest

class BlockDiagTest(RandomTest, MatrixTestBase):

    def setUp(self):
        super().setUp()

        square_ex = [
            (1, 1),
            (2, 1),
            (3, 1),
            (1, 2),
            (2, 3),
            (10, 5)]
        square_ex = [[np.random.rand(size, size) for _ in range(num)]
                     for size, num in square_ex]
        rect_ex = [
            [(1, 2)],
            [(2, 1)],
            [(2, 1), (2, 1)],
            [(2, 2), (1, 2)],
            [(3, 2), (3, 2), (3, 2)],
            [(2, 1), (3, 1), (1, 4)]]
        rect_ex = [[np.random.rand(*shape) for shape in ls] for ls in rect_ex]

        self.raw_examples = square_ex + rect_ex
        self.examples = list(map(self._generate, self.raw_examples))

    @staticmethod
    def _generate(mats):
        my_mats = [NumpyMatrix(x) if not isinstance(x, Matrix) else x
                   for x in mats]
        return BlockDiag(my_mats)

    def test_as_numpy(self):
        for s, ex in zip(self.examples, self.raw_examples):
            np_mat = la.block_diag(*ex)
            np.testing.assert_allclose(s.as_numpy(), np_mat)

    def test_empty(self):
        self.assertRaises(ValueError, BlockDiag, [])
