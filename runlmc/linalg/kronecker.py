# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import numpy as np
import scipy.linalg
import scipy.sparse.linalg

from .matrix import Matrix
from ..util.docs import inherit_doc

@inherit_doc
class Kronecker(Matrix):
    """
    Creates a class with a parsimonious representation of a Kronecker product
    of two :class:`Matrix` instances. For the Kronecker matrix
    :math:`K=A\\otimes B`, the :math:`ij`-th block entry is
    :math:`a_{ij}B`.

    The implementation is based off of Gilboa, Saatçi, and Cunningham (2015).
    """
    def __init__(self, A, B):
        """
        Creates a :class:`Kronecker` matrix.

        :param A: the first matrix
        :param B: the second matrix
        """
        super().__init__(A.shape[0] * B.shape[0])

        self.A = A
        self.B = B

    def matvec(self, x):
        return x

    def eig(self, cutoff):
        return []
