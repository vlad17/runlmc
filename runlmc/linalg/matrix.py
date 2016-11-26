# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import numpy as np
import scipy.sparse.linalg

class Matrix:
    """
    An abstract class defining the interface for the necessary
    sparse matrix operations.

    We only consider square matrices over doubles.
    """

    def __init__(self, n):
        """
        :param n: number of rows in this square matrix
        :raises ValueError: if `n < 1`
        """
        if n < 1:
            raise ValueError('Size of the matrix {} < 1'.format(n))
        self.dtype = np.float64
        self.shape = (n, n)

    def solve(self, b, tol):
        """
        Solves a linear system :math:`A\\textbf{x}=\\textbf{b}` without
        any preconditioners for :math:`\\textbf{x}`. :math:`A` is the
        matrix represented by this class.

        :param b: numpy vector :math:`\\textbf{b}`
        :returns: the linear solution :math:`\\textbf{x}`
        """
        A = scipy.sparse.linalg.aslinearoperator(self)
        cg_solve, success = scipy.sparse.linalg.cg(A, b, tol=tol)
        assert success == 0
        return cg_solve

    def matvec(self, x):
        """
        Multiply a vector :math:`\\textbf{x}` by this matrix,
        :math:`A`, yielding :math:`A\\textbf{x}`.

        :param x: a one-dimensional numpy array of the same size as this matrix
        :returns: the matrix-vector product
        """
        raise NotImplementedError

    def eig(self, cutoff):
        """
        Finds the eigenvalues of this matrix of magnitude above the cutoff.

        :param cutoff: eigenvalue cutoff
        :returns: a numpy array of eigenvalues in decreasing order, repeated by
                  multiplicity.
        """
        raise NotImplementedError
