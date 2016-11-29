# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import numpy as np
import scipy.linalg
import scipy.sparse.linalg

from .psd_matrix import PSDMatrix
from ..util.docs import inherit_doc

@inherit_doc
class Toeplitz(PSDMatrix):
    """
    Creates a class with a parsimonious representation of a PSD
    Toeplitz matrix; that is, a matrix :math:`T` with entries :math:`T_{ij}`
    which for all :math:`i,j` and :math:`i'=i+1, j'=j+1` satisfy:

    .. math::

        t_{ij} = t_{i'j'}

    In addition, :math:`T` must be PSD.

    Notation used in documentation for this class, in addition to its
    implementation, is based on Böttcher, Grudsky, and Maksimenko (2010).
    """
    def __init__(self, top):
        """
        Creates a :class:`Toeplitz` matrix.

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
        circ = self._cyclic_extend(top)
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
        cut = np.searchsorted(sol, cutoff, 'right')
        return sol[cut:][::-1]

    def upper_eig_bound(self):
        # By Gershgorin, we need to find the largest absolute row.
        # This can be computed in linear time with an easy dynamic program,
        # since it's just the largest cyclic permutation of top.
        #
        # The vectorized numpy expression below is equivalent to the following
        # C-like syntax:
        # totals = np.zeros(len(self.top))
        # totals[0] = np.abs(self.top).sum()
        # for i in range(1, len(totals)):
        #     totals[i] = totals[i-1] - self.top[-i] + self.top[i]
        abstop = np.abs(self.top)
        totals = np.copy(abstop)
        totals[0] = abstop.sum()
        totals[1:] -= abstop[:0:-1]
        totals = np.add.accumulate(totals)
        return totals.max()
