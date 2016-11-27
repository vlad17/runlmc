# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import numpy as np
import scipy.linalg
import scipy.sparse.linalg

from .matrix import Matrix
from ..util.docs import inherit_doc

@inherit_doc
class SymmToeplitz(Matrix):
    """
    Creates a class with a parsimonious representation of a symmetric
    Toeplitz matrix; that is, a matrix :math:`T` with entries :math:`T_{ij}`
    which for all :math:`i,j` and :math:`i'=i+1, j'=j+1` satisfy:

    .. math::

        t_{ij} = t_{i'j'}

    Notation used in documentation for this class, in addition to its
    implementation, is based on Böttcher, Grudsky, and Maksimenko (2010).
    """
    def __init__(self, top):
        """
        Creates a :class:`SymmToeplitz` matrix.

        :param top: 1-dimensional :mod:`numpy` array, used as the underlying
                    storage, which represents the first row :math:`t_{1j}`.
                    Should be castable to a float64.
        :raises ValueError: if `top` isn't of the right shape, is empty.
        """
        if top.shape != (len(top),):
            raise ValueError('top shape {} is not 1D'.format(top.shape))
        if len(top) == 0:
            raise ValueError('top is empty')

        super().__init__(len(top))

        self.top = top.astype('float64', casting='safe')
        circ = SymmToeplitz._cyclic_extend(top)
        self.circ_fft = np.fft.fft(circ)

    @staticmethod
    def _cyclic_extend(x):
        n = len(x)
        extended = np.zeros(n * 2)
        extended[:n] = x
        extended[n+1:] = x[1:][::-1]
        return extended

    def matvec(self, x):
        assert len(x) * 2 == len(self.circ_fft)
        x_fft = np.fft.fft(x, n=len(self.circ_fft))
        return np.fft.ifft(self.circ_fft * x_fft)[:len(x)].real

    def eig(self, cutoff):
        # Temporary solution using dense routines.
        # The grudsky implementation from before will be installed shortly
        sol = np.linalg.eigvalsh(scipy.linalg.toeplitz(self.top)).real
        sol = np.sort(sol)
        cut = np.searchsorted(sol, cutoff)
        return sol[cut:][::-1]
