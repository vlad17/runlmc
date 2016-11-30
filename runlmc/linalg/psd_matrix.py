# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import numpy as np
import scipy.sparse.linalg

from ..util.docs import inherit_doc

class PSDMatrix:
    """
    An abstract class defining the interface for the necessary
    sparse matrix operations.

    We consider a very restricted class of matrices only, namely
    positive semi-definite, symmetric, real matrices. In other words,
    the matrix :math:`K` represented by instances of this class is expected to
    adhere to the following semantic laws:

    #. :math:`\\forall\\textbf{x}`,
       :math:`\\textbf{x}^\\top K\\textbf{x} \ge 0`
    #. :math:`K^\\top = K`

    These laws manifest themselves the following property, with
    the actual API, which exposes :func:`matvec` for matrix-vector
    multiplication.

    * Positivity: `K.matvec(x).dot(x) >= 0`
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

@inherit_doc
class PSDDecomposableMatrix(PSDMatrix):
    """
    An extention to a regular :class:`PSDMatrix` which allows for efficient
    eigendecomposition and bounding. This adheres to the following
    mathematical property manifested by the API :func:`eig`.

    * Positive eigenvalues: `len(K.eig(cutoff=0)) == K.shape[0]`
    """

    def __init__(self, n):
        super().__init__(n)

    def eig(self, cutoff):
        """
        Finds the eigenvalues of this matrix of magnitude above the cutoff.

        :param cutoff: eigenvalue cutoff
        :returns: a numpy array of eigenvalues in decreasing order, repeated by
                  multiplicity.
        """
        raise NotImplementedError

    def upper_eig_bound(self):
        """
        Returns an upper bound :math:`B` for the largest eigenvalue of this
        PSD matrix.

        Impementations can rely on the fairly tight and efficient-to-compute
        Gershgorin circle theorem, which implies that the largest eigenvalue
        is bounded by the largest absolute row sum in PSD matices.

        :return: :math:`B`
        """
        raise NotImplementedError
