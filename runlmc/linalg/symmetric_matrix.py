# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import numpy as np
import scipy.sparse.linalg

class SymmetricMatrix:
    """
    An abstract class defining the interface for the necessary
    sparse matrix operations.

    We consider a very restricted class of matrices only, namely
    symmetric, real matrices.

    :param n: number of rows in this square matrix
    :raises ValueError: if `n < 1`
    """

    def __init__(self, n):
        if n < 1:
            raise ValueError('Size of the matrix {} < 1'.format(n))
        self.dtype = np.float64
        self.shape = (n, n)
        self._op = None

    def as_linear_operator(self):
        """
        .. Note:: The :func:`scipy.sparse.linalg.aslinearoperator`
                  converter does not do the same work this does - it doesn't
                  correctly interpret what a symmetric real operator has to
                  offer.

        :returns: this matrix as a
                  :class:`scipy.sparse.linalg.LinearOperator`
        """
        if self._op is None:
            self._op = scipy.sparse.linalg.LinearOperator(
                shape=self.shape,
                dtype=self.dtype,
                matvec=self.matvec,
                rmatvec=self.matvec,
                matmat=self.matmat)
        return self._op

    def as_numpy(self):
        """
        :returns: numpy matrix equivalent, as a 2D :class:`numpy.ndarray`
        """
        raise NotImplementedError

    def matvec(self, x):
        """
        Multiply a vector :math:`\\textbf{x}` by this matrix,
        :math:`K`, yielding :math:`K\\textbf{x}`.

        :param x: a one-dimensional numpy array of the same size as this matrix
        :returns: the matrix-vector product
        """
        raise NotImplementedError

    def matmat(self, X):
        """
        Multiply a matrix :math:`X` by this matrix,
        :math:`K`, yielding :math:`KX`. This just repeatedly calls
        :func:`matvec`.

        :param X: a (possibly rectangular) matrix.
        :returns: the matrix-matrix product
        """
        return np.hstack([self.matvec(col).reshape(-1, 1) for col in X.T])


    def upper_eig_bound(self):
        """
        Impementations can rely on the fairly tight and efficient-to-compute
        Gershgorin circle theorem, which implies that the largest eigenvalue
        is bounded by the largest absolute row sum in PSD matices.
        :return: an upper bound for the largest eigenvalue of this
                 (necessarily diagonalizable) matrix.
        """
        raise NotImplementedError
