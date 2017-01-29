# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import numpy as np

from .symmetric_matrix import SymmetricMatrix
from ..util.docs import inherit_doc
from ..util.numpy_convenience import symm_2d_list_map, begin_end_indices

# TODO(test)
@inherit_doc
class BlockDiag(SymmetricMatrix):
    def __init__(self, blocks):
        self.lens = [block.shape[0] for block in blocks]
        super().__init__(sum(self.lens))
        self.begins, self.ends = begin_end_indices(self.lens)
        self.blocks = blocks

    def matvec(self, x):
        result = np.empty_like(x)
        for begin, end, block in zip(self.begins, self.ends, self.blocks):
            result[begin:end] = block.matvec(x[begin:end])
        return result

    def as_numpy(self):
        z = np.zeros(self.shape)
        for begin, end, block in zip(self.begins, self.ends, self.blocks):
            z[begin:end, begin:end] = block.as_numpy()
        return z

    def upper_eig_bound(self):
        raise NotImplementedError
