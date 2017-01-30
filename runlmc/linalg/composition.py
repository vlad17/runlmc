# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

from .matrix import Matrix
from ..util.docs import inherit_doc

# TODO(test)
@inherit_doc
class Composition(Matrix):
    def __init__(self, mats):
        super().__init__(mats[0].shape[0], mats[-1].shape[1])
        self.mats = mats

    def matvec(self, x):
        for M in reversed(self.mats):
            x = M.matvec(x)
        return x

    def matmat(self, x):
        for M in reversed(self.mats):
            x = M.matmat(x)
        return x
