# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import numpy as np
import scipy.sparse.linalg

from .numpy_matrix import NumpyMatrix
from .psd_matrix import PSDMatrix
from ..util.docs import inherit_doc
from ..util.numpy_convenience import EPS

@inherit_doc
class Kronecker(PSDMatrix):
    """
    Creates a class with a parsimonious representation of a Kronecker product
    of two :class:`Matrix` instances. For the Kronecker matrix
    :math:`K=A\\otimes B`, the :math:`ij`-th block entry is
    :math:`a_{ij}B`.

    :math:`K` is PSD if :math:`A,B` are.

    The implementation is based off of Gilboa, Saatçi, and Cunningham (2015).
    """
    def __init__(self, A, B):
        """
        Creates a :class:`Kronecker` matrix.

        :param A: the first matrix
        :param B: the second matrix
        :raises ValueError: if matrices aren't square
        """
        super().__init__(A.shape[0] * B.shape[0])

        if A.shape[0] != A.shape[1]:
            raise ValueError('A.shape {} not square'.format(A.shape))
        if B.shape[0] != B.shape[1]:
            raise ValueError('B.shape {} not square'.format(B.shape))

        self.A = self._to_mat(A)
        self.B = self._to_mat(B)

    @staticmethod
    def _to_mat(X):
        if isinstance(X, PSDMatrix):
            return X
        elif isinstance(X, np.ndarray):
            return NumpyMatrix(X)
        else:
            raise TypeError('Inputs have to be runlmc.linalg.matrix.Matrix'
                            ' or numpy.ndarray instances')

    def matvec(self, x):
        # This differs from the paper's MVM, but is the equivalent for
        # a C-style ordering of arrays.
        for M in [self.B, self.A]:
            n = M.shape[1]
            x = x.reshape(-1, n).T
            x = M.matmat(x)
        return x.reshape(-1)

    @staticmethod
    def _largest_eig(mat):
        # Gershgorin circle thm
        # TODO: test / make part of interface
        from .toeplitz import Toeplitz
        if isinstance(mat, NumpyMatrix):
            return np.abs(mat.A).sum(axis=0).max()
        elif isinstance(mat, Toeplitz):
            # largest cyclic permutation
            totals = np.zeros(len(mat.top))
            totals[0] = np.abs(mat.top).sum()
            # this can be unfolded into a nonrecursive plus an accumulate
            for i in range(1, len(totals)):
                totals[i] = totals[i-1] - mat.top[-i] + mat.top[i]
            return totals.max()
        else:
            assert isinstance(mat, Kronecker)
            return Kronecker._largest_eig(mat.A)*Kronecker._largest_eig(mat.B)
        return scipy.sparse.linalg.eigsh(
            scipy.sparse.linalg.aslinearoperator(mat),
            k=1,
            which='LA',
            return_eigenvectors=False) # TODO: mess with tol / ncv

    @staticmethod
    def _conservative_cutoff_factor(cutoff, factor):
        return cutoff / factor * (1 - 2 * EPS)

    @staticmethod
    def _binsearch_descending(x, xs):
        # Given a descending array xs, search for the first value below
        # a certain threshold.
        idx = np.searchsorted(xs[::-1], x)
        return len(xs)-idx

    def eig(self, cutoff):
        if self.shape[0] == 1:
            eigs = self.A.eig(0) * self.B.eig(0)
            return eigs if eigs[0] > cutoff else np.array([])

        largeA = self._largest_eig(self.A)
        largeB = self._largest_eig(self.B)
        eigA = self.A.eig(self._conservative_cutoff_factor(cutoff, largeB))
        eigB = self.B.eig(self._conservative_cutoff_factor(cutoff, largeA))
        # Can use smarter filter here - don't need to generate every eigenvalue
        # from the outer product if the smallest is less than the fixed cutoff
        eigs = np.outer(eigA, eigB).reshape(-1)
        # eigs[::-1].sort()[::-1]?
        # mergesort?
        eigs.sort()
        print('large A pred', largeA, 'actual', eigA[0])
        print('large B pred', largeB, 'actual', eigB[0])
        return eigs[np.searchsorted(eigs, cutoff, 'right'):][::-1]
