# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import numpy as np

from .matrix import Matrix
from ..util.docs import inherit_doc

# TODO(test)
@inherit_doc
class Identity(Matrix):
    def __init__(self, n):
        super().__init__(n, n)

    def matvec(self, x):
        return x

    def matmat(self, x):
        return x

    def as_numpy(self):
        return np.identity(self.shape[0])

    def upper_eig_bound(self):
        return 1
