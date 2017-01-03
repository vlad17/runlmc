# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import logging

import numpy as np
import scipy.linalg
import scipy.sparse.linalg

from .psd_matrix import PSDDecomposableMatrix
from ..util.docs import inherit_doc
from ..util.numpy_convenience import search_descending, EPS, smallest_eig

_LOG = logging.getLogger(__name__)

@inherit_doc
class Toeplitz(PSDDecomposableMatrix):
    """
    Creates a class with a parsimonious representation of a PSD
    Toeplitz matrix; that is, a matrix :math:`T` with entries :math:`T_{ij}`
    which for all :math:`i,j` and :math:`i'=i+1, j'=j+1` satisfy:

    .. math::

        t_{ij} = t_{i'j'}

    In addition, :math:`T` must be PSD.

    The :func:`eig` implementation assumes an ordering and positivity
    :math:`0\le t_{1i}\le t_{1j}` for :math:`i\ge j`.

    :param top: 1-dimensional :mod:`numpy` array, used as the underlying
                storage, which represents the first row :math:`t_{1j}`.
                Should be castable to a float64.
    :raises ValueError: if `top` isn't of the right shape or is empty.
    :raises RuntimError: if induced Toeplitz matrix
                         is not PSD (if logger with this module's fully
                         qualified name is set to debug mode)
    """

    def __init__(self, top):
        if top.shape != (len(top),):
            raise ValueError('top shape {} is not 1D'.format(top.shape))
        if len(top) == 0:
            raise ValueError('top is empty')

        super().__init__(len(top))

        self.top = top.astype('float64', casting='safe')
        circ = self._cyclic_extend(top)
        self._circ_fft = np.fft.fft(circ)

        if _LOG.isEnabledFor(logging.DEBUG):
            sm = smallest_eig(self.top)
            cutoff = len(top) ** 2 * EPS
            if sm < -cutoff:
                raise RuntimeError('Eigenvalue {} below tolerance {}\n{!s}'
                                   .format(sm, -cutoff, self))

    @staticmethod
    def _cyclic_extend(x):
        n = len(x)
        extended = np.zeros(n * 2)
        extended[:n] = x
        extended[n+1:] = x[1:][::-1]
        return extended

    def as_numpy(self):
        return scipy.linalg.toeplitz(self.top)

    def matvec(self, x):
        # This MVM takes advantage of a well-known circulant embedding
        # of this Toeplitz matrix, enabling O(n lg n) multiplication
        # By embedding the Toeplitz matrix into the upper-left quadrant
        # of a cyclically-extended circulant, the product of a zero-extended
        # vector with the circulant matrix (a circular convolution, and
        # therefore Fourier product) gives the Toeplitz-vector product
        # in the first half of the result.
        assert len(x) * 2 == len(self._circ_fft)
        x_fft = np.fft.fft(x, n=len(self._circ_fft))
        return np.fft.ifft(self._circ_fft * x_fft)[:len(x)].real

    def _dense_eig(self, cutoff):
        # Default exact solution
        # Ng and Trench 1997 O(n^2) per-eigenvalue approach for this exists
        # enabling O(n^3 / p), with p the parallelism
        # Böttcher, Grudsky, and Maksimenko 2010 approach is also accurate
        # and is O(r^3n) per eigenvalue, where r is depedent on how
        # diagonally dominant the matrix is. This can analogously be parallel.
        sol = np.linalg.eigvalsh(self.as_numpy()).real
        sol[::-1].sort()
        return sol[:search_descending(cutoff, sol, inclusive=False)]

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

    def eig(self, cutoff, exact):
        if exact:
            return self._dense_eig(cutoff)

        assert np.all(self.top) >= 0

        # Should have some kind of condition number check here...

        # This approach is based on Böttcher, Grudsky, and Maksimenko 2010.
        #
        # The authors provide an iteration routine that runs in time
        # O(n w^3), where w is the "effective width" of the Toeplitz matrix,
        # the number of entries whose magnitude is no less than cutoff times
        # the largest element (which is along the main diagonal).
        #
        # The first iteration in this routine, however, has a nicer form
        # that can be computed in O(n lg n) time, independent of w.
        #
        # To briefly summarize what's going on: a symmetric Toeplitz matrix
        # of size n induces a Laurent polynomial a.
        #
        # If we take c_j to be the elements of a symmetric Toeplitz
        # matrix's first column, and set c_{-j}=c_j, then for the
        # vector c and Laurent basis from -n to n and t of entries t^j,
        # with again j spanning from -n to n: a = c . t.
        # This is real-valued on the unit complex circle, T.
        #
        # We pull back a's domain from T to the interval
        # [0, pi] with g(x) = a(exp(ix))
        #
        # Theorem 1.2 of the Grudsky 2010 paper states that the solution
        # (n + 1) phi(lambda) + theta(lambda) = pi * j
        # determines the j-th largest eigenvalue of our Toeplitz matrix
        # up to a factor exponentially small in n, for certain functions
        # phi and theta.
        #
        # It is shown phi^-1 = psi = g. This returns psi applied to each
        # value in lambdas. Since it is real-valued, we only concern
        # ourselves with the real terms.
        #
        # The iteration from section 4 defines a new iterate lambda' from
        # a starting guess lambda by solving the formula from Theorem 1.2:
        #
        # lambda' = phi^-1((pi * j - theta(lambda)) / (n + 1))
        #
        # If the initial guess for each eigenvalue is 0, the above,
        # vectorized over j, is equivalent to the Fourier transform of
        # of c.

        # TODO(MSGP) approximation; use whittle (?)

        n = len(self.top)
        N = 2 * n + 2

        # For tridiagonal matrices, Grudsky mentions this is exact.
        lam = 2 * np.fft.fft(self.top, n=N)[1:][:n].real - self.top[0]
        lam[::-1].sort()

        return lam[:search_descending(cutoff, lam, inclusive=False)]
