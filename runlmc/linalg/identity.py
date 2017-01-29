# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import numpy as np

from .symmetric_matrix import SymmetricMatrix
from ..util.docs import inherit_doc

# TODO(test)
@inherit_doc
class Identity(SymmetricMatrix):
    def __init__(self, n):
        super().__init__(n)

    def matvec(self, x):
        return x

    def matmat(self, x):
        return x

    def as_numpy(self):
        return np.identity(self.shape[0])

    def upper_eig_bound(self):
        return 1
