# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import numpy as np
import scipy.linalg
import scipy.sparse.linalg

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

        top = top.astype('float64', casting='safe')

        circ = SymmToeplitz._cyclic_extend(top)
        self.circ_fft = np.fft.fft(circ)

        self.dtype = top.dtype
        self.shape = (len(top), len(top))
        self.top = top

    @staticmethod
    def _cyclic_extend(x):
        n = len(x)
        extended = np.zeros(n * 2)
        extended[:n] = x
        extended[n+1:] = x[1:][::-1]
        return extended

    def matvec(self, x):
        """
        Multiply a vector by this matrix. This implementation uses a Circulant
        embedding and an FFT to compute the product in :math:`O(n\log n)` time.

        :param x: a one-dimensional numpy array of the same size as this matrix
        :returns: the matrix-vector product
        """
        assert len(x) * 2 == len(self.circ_fft)
        x_fft = np.fft.fft(x, n=len(self.circ_fft))
        return np.fft.ifft(self.circ_fft * x_fft)[:len(x)].real


    def solve(self, b, tol):
        """
        Solves a linear system :math:`A\textbf{x}=\textbf{b}` without
        any preconditioners for :math:`\textbf{x}`. :math:`A` is the
        matrix represented by this class.

        :param b: numpy vector :math:`\textbf{b}`
        :returns: the linear solution :math:`\textbf{x}`
        """
        A = scipy.sparse.linalg.aslinearoperator(self)
        cg_solve, success = scipy.sparse.linalg.cg(A, b, tol=tol)
        assert success == 0
        return cg_solve

    def eig(self, tol):
        """
        Finds the eigenvalues of this matrix of magnitude above the cutoff.

        :param tol: eigenvalue tolerance size
        :returns: a numpy array of eigenvalues in decreasing order, repeated by
                  multiplicity.
        """
        A = scipy.sparse.linalg.aslinearoperator(self)
        N = len(self.top)
        import math
        k = max(math.log(N), 16)
        sol = None
        while k < N:
            sol = scipy.sparse.linalg.eigsh(
                A,
                k=k,
                which='LM',
                return_eigenvectors=False,
                tol=(tol / 3))
            # Any imaginary components are due to round-off, since we're
            # Hermitian
            sol = sol.real
            if sol.min() <= tol:
                break
            k *= 2
        if k >= N:
            sol = np.linalg.eigvalsh(scipy.linalg.toeplitz(self.top))
        sol = np.sort(sol)
        cut = np.searchsorted(sol, tol)
        return sol[cut:][::-1]
