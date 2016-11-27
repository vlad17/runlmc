# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import numpy as np
import scipy.linalg
import scipy.sparse.linalg

from .matrix import Matrix
from ..util.docs import inherit_doc
from ..util.numpy_convenience import EPS

@inherit_doc
class SymmNpMatrix(Matrix):
    """
    Adapter to :class:`Matrix` with :mod:`numpy` arrays.
    """
    def __init__(self, nparr):
        """
        Creates a :class:`SymmNpMatrix` matrix.

        :param nparr: 2-dimensional :mod:`numpy` array
        :raises ValueError: if `nparr` isn't 2D or square or symmetric
        """
        if nparr.ndim != 2 or nparr.shape[0] != nparr.shape[1]:
            raise ValueError('Input numpy array of shape {} not square matrix'
                             .format(nparr.shape))

        self.A = nparr.astype('float64', casting='safe')

        if not np.allclose(self.A, self.A.T, rtol=0, atol=(EPS * 2)):
            raise ValueError('Input numpy array not symmetric')

        super().__init__(len(nparr))

    def matvec(self, x):
        return self.A.dot(x)

    def eig(self, cutoff):
        sol = np.linalg.eigvalsh(self.A).real
        sol = np.sort(sol)
        cut = np.searchsorted(sol, cutoff)
        return sol[cut:][::-1]

    def test_shape_2d(self): # TODO move to test
        A = np.arange(3 * 4 * 5).reshape(3, 4, 5)
        self.assertRaises(ValueError, SymmNpMatrix, A)
        self.assertRaises(ValueError, SymmNpMatrix, A.reshape(-1))

    # nonsymm test
