# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import numpy as np
import scipy.linalg as la

from .matrix import Matrix
from ..util.docs import inherit_doc
from ..util.numpy_convenience import begin_end_indices

@inherit_doc
class BlockDiag(Matrix):
    """
    Creates a block diagonal matrix from constintuent, possibly
    rectangular internal ones. Note this is PSD if its constintuents are.

    For constituents :math:`K_i`, this matrix corresponds to the direct sum
    :math:`K = \\bigoplus_i K_i`.

    :param blocks: blocks with which to construct the block diagonal matrix.
    """

    def __init__(self, blocks):
        row_lens = [block.shape[0] for block in blocks]
        col_lens = [block.shape[1] for block in blocks]
        super().__init__(sum(row_lens), sum(col_lens))
        self.rbegins, self.rends = begin_end_indices(row_lens)
        self.cbegins, self.cends = begin_end_indices(col_lens)
        self.blocks = blocks

    def _iterate_all(self):
        return zip(self.rbegins, self.rends,
                   self.cbegins, self.cends,
                   self.blocks)

    def matvec(self, x):
        result = np.empty(self.shape[0], dtype=self.dtype)
        for rbegin, rend, cbegin, cend, block in self._iterate_all():
            result[rbegin:rend] = block.matvec(x[cbegin:cend])
        return result

    def as_numpy(self):
        return la.block_diag(*(block.as_numpy() for block in self.blocks))

    def __str__(self):
        return ('BlockDiag(..., blocki, ...)\n' +
                '\n'.join(
                    ['block{}\n{!s}'.format(i, block)
                     for i, block in enumerate(self.blocks)]))
