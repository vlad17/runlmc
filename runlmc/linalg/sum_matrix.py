# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import numpy as np

from .psd_matrix import PSDMatrix
from ..util.docs import inherit_doc

@inherit_doc
class SumMatrix(PSDMatrix):
    """
    The sum matrix represents a sum of other possibly sparse
    :class:`runlmc.linalg.PSDDecomposableMatrix` instances :math:`A_i`; i.e.,
    it takes on the following meaning:

    .. math::

        \DeclareMathOperator{\diag}{diag}
        \sum_iA_i+\diag\\boldsymbol\\epsilon

    It is special in that it does not admit an easy way to
    eigendecompose itself given the eigendecompostion of its elements.

    However, it has some other convenient properties.

    :param Ks: decomposable matrices to sum
    :param noise: :math:`\\boldsymbol\epsilon`
    :raises ValueError: if any elements of `noise < 0`
    :raises ValueError: if `len(noise)` isn't equal to matrix size
    """

    def __init__(self, Ks, noise):
        if len(Ks) > 0:
            shapes = [K.shape for K in Ks]
            if len(set(shapes)) != 1:
                raise ValueError('At most one distinct shape expected in sum, '
                                 'found shapes:\n{}'.format(shapes))
            n = shapes[0][0]
        else:
            n = len(noise)

        if len(noise) != n:
            raise ValueError('noise length {} != size {}'
                             .format(len(noise), n))
        if np.any(noise < 0):
            raise ValueError('noise must be nonnegative')

        super().__init__(n)
        self.Ks = Ks
        self.noise = noise

    def matvec(self, x):
        return sum(K.matvec(x) for K in self.Ks) + self.noise * x

    def __str__(self):
        return (
            'SumMatrix([..., Ki, ...]) + noise\n' +
            'noise\n' +
            str(self.noise) + '\n' +
            '\n'.join(
                ['K{}\n{!s}'.format(i, K) for i, K in enumerate(self.Ks)]))

    def logdet(self):
        """
        Using the eigendecomposition of each :math:`A_i`, this method computes
        an **upper bound** of the log-determinant:

        .. math::

            \log\det\\left|K\\right|

        This is done by approximately solving an optimization problem.
        For details, and an elementary proof of the solution, see [TODO PAPER].
        """
        # separate into a minimum constant diagonal perturbation
        # plus a sum of matrices
        Q = len(self.Ks)
        n = self.shape[0]
        min_err = max(self.noise.min(), 1e-10)
        noise = np.copy(self.noise)
        noise[noise < min_err] = min_err
        eigs = np.zeros((Q + 1, n)) + min_err
        eigs[-1] = noise
        for i, K in enumerate(self.Ks):
            eigs_K = K.eig(min_err)
            eigs[i, :len(eigs_K)] = eigs_K

        eigsT = eigs.T
        for i in range(Q):
            eigsT = eigsT[np.argsort(eigsT[:, :i].sum(-1))]
            eigsT[:, i][::-1].sort()
        eigs = eigsT.T

        return np.log(eigs.sum(axis=0)).sum()
