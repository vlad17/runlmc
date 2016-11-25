# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import numpy as np
import scipy.linalg

class SymmToeplitz:
    """
    Creates a class with a parsimonious representation of a symmetric
    Toeplitz matrix; that is, a matrix :math:`T` with entries :math:`T_ij`
    which for all :math:`i,j` and :math:`i'=i+1, j'=j+1` satisfy:

    .. math::

        t_{ij} = t_{i'j'}

    It admits an :math:`O(n)` space representation and can compute
    matrix-vector products and approximate eigenvalues using
    as-much additional space and in :math:`O(n \log n)` time.
    """
    def __init__(self, top):
        """
        Creates a :class:`SymmToeplitz` matrix.

        :param top: 1-dimensional :mod:`numpy` array, used as the underlying
                    storage, which represents the first row :math:`t_{1j}`.
        :raises ValueError: if `top` isn't of the right shape or is empty.
        """
        if top.shape != (len(top),):
            raise ValueError('top shape {} is not 1D'.format(top.shape))
        if len(top) == 0:
            raise ValueError('top is empty')

        circ = SymmToeplitz._cyclic_extend(top)
        self.circ_fft = np.fft.fft(circ)

        self.dtype = self.circ_fft.dtype
        self.shape = (len(top), len(top))

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
        return np.fft.ifft(self.circ_fft * x_fft)[:len(x)]
