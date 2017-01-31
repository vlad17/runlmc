# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import numpy as np

from .test_matrix_base import MatrixTestBase
from .numpy_matrix import NumpyMatrix
from .block_matrix import SymmSquareBlockMatrix
from ..util.testing_utils import RandomTest
from ..util.numpy_convenience import symm_2d_list_map

class SymmSquareBlockTest(RandomTest, MatrixTestBase):

    def setUp(self):
        super().setUp()

        def symm(sz):
            x = np.random.rand(sz, sz)
            return x + x.T

        def symm_symm(sz, num):
            block = np.array([[symm(sz)
                               for _ in range(num)]
                              for _ in range(num)])
            for i in range(len(block)):
                for j in range(i):
                    block[i, j] = block[j, i]
            return block

        square_ex = [
            (1, 1),
            (2, 1),
            (3, 1),
            (1, 2),
            (2, 3),
            (10, 5)]

        self.raw_examples = [symm_symm(size, num)
                             for size, num in square_ex]
        self.examples = list(map(self._generate, self.raw_examples))

    @staticmethod
    def _generate(mats):
        my_mats = symm_2d_list_map(NumpyMatrix, mats, len(mats))
        return SymmSquareBlockMatrix(my_mats)

    def test_as_numpy(self):
        for s, ex in zip(self.examples, self.raw_examples):
            np_mat = np.bmat([[ex[i, j] for j in range(ex.shape[1])]
                              for i in range(ex.shape[0])]).A
            np.testing.assert_allclose(s.as_numpy(), np_mat)

    def test_empty(self):
        self.assertRaises(ValueError, SymmSquareBlockMatrix, [[]])
        self.assertRaises(ValueError, SymmSquareBlockMatrix, [[]])
