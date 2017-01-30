# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

from .matrix import Matrix
from ..util.docs import inherit_doc

@inherit_doc
class NumpyMatrix(Matrix):
    """
    Adapter to :class:`Matrix` with :mod:`numpy` arrays.

    Creates a :class:`NumpyMatrix` matrix.

    :param nparr: 2-dimensional :mod:`numpy` array
    :raises ValueError: if `nparr` isn't 2D
    """
    def __init__(self, nparr):
        if nparr.ndim != 2:
            raise ValueError('Input numpy array of shape {} not matrix'
                             .format(nparr.shape))
        self.A = nparr.astype('float64', casting='safe')
        super().__init__(*self.A.shape)

    def as_numpy(self):
        return self.A

    def matvec(self, x):
        return self.A.dot(x)

    def matmat(self, x):
        return self.A.dot(x)

    def __str__(self):
        return str(self.A)
