# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import numpy as np

from .matrix import Matrix
from ..util.docs import inherit_doc
from ..util.numpy_convenience import symm_2d_list_map

@inherit_doc
class SymmSquareBlockMatrix(Matrix):
    """
    Creates a block matrix from a 2D array of matrices, which must
    all be square. The blocks array is assumed to be symmetric.
    :param blocks: a 2D array
    :raises ValueError: if the size is 0.
    """
    def __init__(self, blocks):
        self.D = len(blocks)
        if set(map(len, blocks)) != {self.D}:
            raise ValueError('Uneven sizes')
        m = blocks[0][0].shape[0]
        n = self.D * m
        super().__init__(n, n)
        self.blocks = blocks
        self.begins = np.arange(0, self.shape[0], m)
        self.ends = self.begins + m

    def matvec(self, x):
        result = np.zeros_like(x, dtype=self.dtype)
        for rbegin, rend, row in zip(self.begins, self.ends, self.blocks):
            for cbegin, cend, block in zip(self.begins, self.ends, row):
                result[rbegin:rend] += block.matvec(x[cbegin:cend])
        return result

    def as_numpy(self):
        z = np.zeros(self.shape)
        for rbegin, rend, row in zip(self.begins, self.ends, self.blocks):
            for cbegin, cend, block in zip(self.begins, self.ends, row):
                z[rbegin:rend, cbegin:cend] = block.as_numpy()
        return z

    def upper_eig_bound(self):
        bounds = symm_2d_list_map(lambda x: x.upper_eig_bound(),
                                  self.blocks, self.D, dtype=float)
        return np.linalg.norm(bounds, 1)

    def __str__(self):
        return ('SymmBlockMatrix(..., block(i,j), ...)\n' +
                '\n'.join(['block({},{})\n{!s}'.format(i, j, self.blocks[i, j])
                           for i in range(self.D) for j in range(i, self.D)]))
