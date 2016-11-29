# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import numpy as np
import scipy.sparse.linalg

class PSDMatrix:
    """
    An abstract class defining the interface for the necessary
    sparse matrix operations.

    We consider a very restricted class of matrices only, namely
    positive semi-definite, symmetric, real matrices. In other words,
    the matrix :math:`A` represented by instances of this class is expected to
    adhere to the following semantic laws:

    #. :math:`\\forall\\textbf{x}`, :math:`\\textbf{x}^\\top A\\textbf{x} \ge 0`
    #. :math:`A^\\top = A`

    These laws manifest themselves through the following properties, through
    the actual API, which exposes :func:`matvec` for matrix-vector
    multiplication and :func:`eig` for eigenvalues.

    #. `A.matvec(x).dot(x) >= 0`
    #. `len(A.eig(cutoff=0)) == A.shape[0]`
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

    def as_linear_operator(self):
        """
        .. Note:: The :func:`scipy.sparse.linalg.aslinearoperator`
                  converter does not do the same work this does - it doesn't
                  correctly interpret what a PSD operator has to offer.

        :returns: this matrix as a
                  :class:`scipy.sparse.linalg.LinearOperator`
        """
        return scipy.sparse.linalg.LinearOperator(
            shape=self.shape,
            dtype=self.dtype,
            matvec=self.matvec,
            rmatvec=self.matvec,
            matmat=self.matmat)

    def matvec(self, x):
        """
        Multiply a vector :math:`\\textbf{x}` by this matrix,
        :math:`A`, yielding :math:`A\\textbf{x}`.

        :param x: a one-dimensional numpy array of the same size as this matrix
        :returns: the matrix-vector product
        """
        raise NotImplementedError

    def matmat(self, X):
        """
        Multiply a matrix :math:`X` by this matrix,
        :math:`A`, yielding :math:`AX`. This just repeatedly calls
        :func:`matvec`.

        :param X: a (possibly rectangular) matrix.
        :returns: the matrix-matrix product
        """
        return np.hstack([self.matvec(col).reshape(-1, 1) for col in X.T])

    def eig(self, cutoff):
        """
        Finds the eigenvalues of this matrix of magnitude above the cutoff.

        :param cutoff: eigenvalue cutoff
        :returns: a numpy array of eigenvalues in decreasing order, repeated by
                  multiplicity.
        """
        raise NotImplementedError
