# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import numpy as np

from .symmetric_matrix import SymmetricMatrix
from ..util.docs import inherit_doc
from ..util.numpy_convenience import EPS

@inherit_doc
class NumpyMatrix(SymmetricMatrix):
    """
    Adapter to :class:`SymmetricMatrix` with :mod:`numpy` arrays.

    Creates a :class:`NumpyMatrix` matrix. Its positivity is assumed.

    :param nparr: 2-dimensional :mod:`numpy` array
    :raises ValueError: if `nparr` isn't 2D or square or symmetric
    """
    def __init__(self, nparr):
        if nparr.ndim != 2 or nparr.shape[0] != nparr.shape[1]:
            raise ValueError('Input numpy array of shape {} not square matrix'
                             .format(nparr.shape))

        self.A = nparr.astype('float64', casting='safe')

        if not np.allclose(self.A, self.A.T, rtol=EPS, atol=(EPS * 2)):
            raise ValueError('Input numpy array {} not symmetric '
                             .format(self.A.shape))

        super().__init__(len(nparr))

    def as_numpy(self):
        return self.A

    def matvec(self, x):
        return self.A.dot(x)

    def upper_eig_bound(self):
        return np.abs(self.A).sum(axis=1).max()

    def __str__(self):
        return str(self.A)
