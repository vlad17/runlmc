# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import logging

import numpy as np
import scipy.sparse.linalg

from .numpy_matrix import NumpyMatrix
from .psd_matrix import PSDMatrix
from ..util.docs import inherit_doc
from ..util.numpy_convenience import EPS

_LOG = logging.getLogger(__name__)

@inherit_doc
class Kronecker(PSDMatrix):
    """
    Creates a class with a parsimonious representation of a Kronecker product
    of two :class:`runlmc.linalg.psd_matrix.PSDMatrix` instances.
    For the Kronecker matrix
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
            raise TypeError('Inputs have to be '
                            'runlmc.linalg.psd_matrix.PSDMatrix'
                            ' or numpy.ndarray instances')

    def matvec(self, x):
        # This differs from the paper's MVM, but is the equivalent for
        # a C-style ordering of arrays.
        for M in [self.B, self.A]:
            n = M.shape[1]
            x = x.reshape(-1, n).T
            x = M.matmat(x)
        return x.reshape(-1)

    def upper_eig_bound(self):
        return self.A.upper_eig_bound() * self.B.upper_eig_bound()

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

        largeA = self.A.upper_eig_bound()
        largeB = self.B.upper_eig_bound()
        cutoffA = self._conservative_cutoff_factor(cutoff, largeB)
        cutoffB = self._conservative_cutoff_factor(cutoff, largeA)

        _LOG.info('%s eig(cutoff=%8.4g) -> A %s eig(cutoff=%8.4g)',
                  self.shape, cutoff, self.A.shape, cutoffA)
        eigA = self.A.eig(cutoffA)
        _LOG.info('%s A %s largest eig predicted %8.4g actual %8.4g',
                  self.shape, self.A.shape, largeA,
                  eigA[0] if len(eigA) > 0 else 0)

        _LOG.info('%s eig(cutoff=%8.4g) -> B %s eig(cutoff=%8.4g)',
                  self.shape, cutoff, self.B.shape, cutoffB)
        eigB = self.B.eig(cutoffB)
        _LOG.info('%s B %s largest eig predicted %8.4g actual %8.4g',
                  self.shape, self.B.shape, largeB,
                  eigB[0] if len(eigB) > 0 else 0)

        # Can use smarter filter here - don't need to generate every eigenvalue
        # from the outer product if the smallest is less than the fixed cutoff
        eigs = np.outer(eigA, eigB).reshape(-1)
        # eigs[::-1].sort()[::-1]?
        # mergesort?
        eigs.sort()
        return eigs[np.searchsorted(eigs, cutoff, 'right'):][::-1]

    def __str__(self):
        return 'Kron(A, B)\nA\n{!s}\nB\n{!s}'.format(self.A, self.B)
