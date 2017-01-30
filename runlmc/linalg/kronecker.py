# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import logging

import numpy as np

from .matrix import Matrix
from ..util.docs import inherit_doc

_LOG = logging.getLogger(__name__)

@inherit_doc
class Kronecker(Matrix):
    """
    Creates a class with a parsimonious representation of a Kronecker product
    of two :class:`runlmc.linalg.matrix.Matrix` instances.
    For the Kronecker matrix
    :math:`K=A\\otimes B`, the :math:`ij`-th block entry is
    :math:`a_{ij}B`.

    :math:`K` is PSD if :math:`A,B` are.

    The implementation is based off of Gilboa, Saatçi, and Cunningham (2015).

    Creates a :class:`Kronecker` matrix.

    :param A: the first matrix
    :param B: the second matrix
    """
    def __init__(self, A, B):
        super().__init__(A.shape[0] * B.shape[0], A.shape[1] * B.shape[1])
        self.A = A
        self.B = B

    def as_numpy(self):
        return np.kron(self.A.as_numpy(), self.B.as_numpy())

    def matvec(self, x):
        # This differs from the paper's MVM, but is the equivalent for
        # a C-style ordering of arrays.
        for M in [self.B, self.A]:
            n = M.shape[1]
            x = x.reshape(-1, n).T
            x = M.matmat(x)
        return x.reshape(-1)

    def __str__(self):
        return 'Kron(A, B)\nA\n{!s}\nB\n{!s}'.format(self.A, self.B)

    def upper_eig_bound(self):
        return self.A.upper_eig_bound() * self.B.upper_eig_bound()
