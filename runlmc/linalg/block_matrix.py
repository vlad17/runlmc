# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import numpy as np

from .symmetric_matrix import SymmetricMatrix
from ..util.docs import inherit_doc
from ..util.numpy_convenience import symm_2d_list_map

# TODO(test)
@inherit_doc
class BlockMatrix(SymmetricMatrix):
    def __init__(self, blocks):
        m = blocks[0][0].shape[0]
        self.D = len(blocks)
        super().__init__(self.D * m)
        self.blocks = blocks
        self.begins = np.arange(0, self.shape[0], m)
        self.ends = self.begins + m

    def matvec(self, x):
        result = np.zeros_like(x)
        for rbegin, rend, row in zip(self.begins, self.ends, self.blocks):
            for cbegin, cend, block in zip(self.begins, self.ends, row):
                result[rbegin:rend] += block.matvec(x[cbegin:cend])
        return result

    def as_numpy(self):
        mats = symm_2d_list_map(lambda x: x.as_numpy(), self.blocks, self.D)
        return np.bmat(mats).A

    def upper_eig_bound(self):
        bounds = symm_2d_list_map(lambda x: x.upper_eig_bound(),
                                  self.blocks, self.D, dtype=float)
        bounds = bounds.astype(float)
        return np.linalg.norm(bounds, 1)
