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
