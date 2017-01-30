# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import numpy as np
import scipy.sparse.linalg

class Matrix:
    """
    An abstract class defining the interface for the necessary
    sparse matrix operations.

    All matrices are assumed real.

    :param n: number of rows in this matrix
    :param m: number of columns in this matrix
    :raises ValueError: if `n < 1 or m < 1`
    """

    def __init__(self, n, m):
        if n < 1 or m < 1:
            raise ValueError('Size of the matrix {} < 1'.format((n, m)))
        self.dtype = np.float64
        self.shape = (n, m)
        self._op = None

    def as_linear_operator(self):
        """
        :returns: this matrix as a
                  :class:`scipy.sparse.linalg.LinearOperator`
        """
        if self._op is None:
            self._op = scipy.sparse.linalg.LinearOperator(
                shape=self.shape,
                dtype=self.dtype,
                matvec=self.matvec,
                matmat=self.matmat)
        return self._op

    def as_numpy(self):
        """
        :returns: numpy matrix equivalent, as a 2D :class:`numpy.ndarray`
        """
        return self.matmat(np.identity(self.shape[1]))

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
        :math:`K`, yielding :math:`KX`. By default, this just repeatedly calls
        :func:`matvec`.

        :param X: a (possibly rectangular) dense matrix.
        :returns: the matrix-matrix product
        """
        result = np.empty(shape=(X.shape[1], self.shape[0]))
        for i, col in enumerate(X.T):
            result[i] = self.matvec(col)
        return result.T

    def is_square(self):
        return self.shape[0] == self.shape[1]

    @staticmethod
    def wrap(shape, mvm):
        return _MatrixImpl(shape, mvm)

class _MatrixImpl(Matrix):
    def __init__(self, shape, mvm):
        super().__init__(*shape)
        self._mvm = mvm

    def matvec(self, x):
        return self._mvm(x)
