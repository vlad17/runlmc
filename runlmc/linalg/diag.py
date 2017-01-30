# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import numpy as np

from .matrix import Matrix
from ..util.docs import inherit_doc

@inherit_doc
class Diag(Matrix):
    """
    Creates a diagonal matrix.
    :param v: main diagaonal
    :raises ValueError: if v is not a non-empty vector
    """

    def __init__(self, v):
        super().__init__(len(v), len(v))
        if v.ndim != 1:
            raise ValueError('Expected input vector for Diagonal matrix '
                             'go something of shape {}'.format(v))
        self.v = v

    def matvec(self, x):
        return x * self.v

    def matmat(self, x):
        # Multiplication by a diagonal matrix on the left corresponds to
        # scaling the rows
        return self.v.reshape(-1, 1) * x

    def as_numpy(self):
        return np.diag(self.v)

    def __str__(self):
        return 'Diag(len {}): {}'.format(len(self.v), self.v)

    def upper_eig_bound(self):
        return self.v.max()
