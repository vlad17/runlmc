# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import logging

import numpy as np
import scipy.linalg as la

from .matrix import Matrix
from ..util.docs import inherit_doc
from ..util.numpy_convenience import EPS

_LOG = logging.getLogger(__name__)


@inherit_doc
class Toeplitz(Matrix):
    """
    Creates a class with a parsimonious representation of a symmetric,
    square
    Toeplitz matrix; that is, a matrix :math:`T` with entries :math:`T_{ij}`
    which for all :math:`i,j` and :math:`i'=i+1, j'=j+1` satisfy:

    .. math::

        t_{ij} = t_{i'j'}

    :param top: 1-dimensional :mod:`numpy` array, used as the underlying
                storage, which represents the first row :math:`t_{1j}`.
                Should be castable to a float64.
    :raises ValueError: if `top` isn't of the right shape or is empty.
    """

    def __init__(self, top):
        if top.shape != (len(top),):
            raise ValueError('top shape {} is not 1D'.format(top.shape))
        if not top.size:
            raise ValueError('top is empty')

        super().__init__(len(top), len(top))

        self.top = top.astype('float64', casting='safe')
        circ = self._cyclic_extend(top)
        self._circ_fft = np.fft.rfft(circ)

    @staticmethod
    def _cyclic_extend(x):
        n = len(x)
        extended = np.zeros(n * 2)
        extended[:n] = x
        extended[n + 1:] = x[1:][::-1]
        return extended

    def as_numpy(self):
        return la.toeplitz(self.top)

    def matvec(self, x):
        # This MVM takes advantage of a well-known circulant embedding
        # of this Toeplitz matrix, enabling O(n lg n) multiplication
        # By embedding the Toeplitz matrix into the upper-left quadrant
        # of a cyclically-extended circulant, the product of a zero-extended
        # vector with the circulant matrix (a circular convolution, and
        # therefore Fourier product) gives the Toeplitz-vector product
        # in the first half of the result.
        x_fft = np.fft.rfft(x, n=(len(x) * 2))
        x_fft *= self._circ_fft
        return np.fft.irfft(x_fft)[:len(x)]

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
        return totals.max() * (1 + EPS * len(self.top))

    def __str__(self):
        if len(self.top) > 10:
            topstr = 'size {}'.format(len(self.top))
        else:
            topstr = str(self.top)
        return 'Toeplitz ' + topstr
