# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

from .matrix import Matrix
from ..util.docs import inherit_doc

@inherit_doc
class SumMatrix(Matrix):
    """
    The sum matrix represents a sum of other possibly sparse
    :class:`runlmc.linalg.SymmetricMatrix` instances :math:`A_i`,
    taking on the meaning :math:`\\sum_iA_i`.

    :param Ks: decomposable matrices to sum
    :raises ValueError: If `Ks` is empty
    """

    def __init__(self, Ks):
        if len(Ks) == 0:
            raise ValueError('Need at least one matrix to sum')

        shapes = [K.shape for K in Ks]
        if len(set(shapes)) != 1:
            raise ValueError('At most one distinct shape expected in sum, '
                             'found shapes:\n{}'.format(shapes))

        super().__init__(*shapes[0])
        self.Ks = Ks

    def matvec(self, x):
        return sum(K.matvec(x) for K in self.Ks)

    def as_numpy(self):
        return sum(K.as_numpy() for K in self.Ks)

    def __str__(self):
        return (
            'SumMatrix([..., Ki, ...])\n' +
            '\n'.join(
                ['K{}\n{!s}'.format(i, K) for i, K in enumerate(self.Ks)]))

    def upper_eig_bound(self):
        # Due to Weyl: very rough bound
        return sum(K.upper_eig_bound() for K in self.Ks)
