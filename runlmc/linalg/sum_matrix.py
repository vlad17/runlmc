# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import numpy as np

from .psd_matrix import PSDMatrix
from ..util.docs import inherit_doc

@inherit_doc
class SumMatrix(PSDMatrix):
    """
    The sum matrix represents a sum of other possibly sparse
    :class:`runlmc.linalg.PSDDecomposableMatrix` instances :math:`A_i`,
    taking on the meaning :math:`\\sum_iA_i`.

    It is special in that it does not admit an easy way to
    eigendecompose itself given the eigendecompostion of its elements.

    However, it has some other convenient properties.

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

        super().__init__(shapes[0][0])
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

    APPROX_EIG_METHODS = [
        'greedy_fiedler',
        'simultaneous_diag',
        'greedy_weyl']

    def approx_eigs(self, min_eig, method='greedy_fiedler'):
        """
        Using the eigendecomposition of each :math:`A_i`, this method computes
        a set of psuedo-eigenvalues of :math:`K`. If multiplied (or,
        to avoid overflow, log-summed), the result is an approximation
        to the log-determinant:

        .. math::

            \log\\left|K\\right|

        None of these methods have any particular guarantees. The
        approximations are done by either minimizing an upper bound
        or maximizing a lower bound.

        The current empirically best strategy, `'greedy_fiedler'`, is
        a greedy calculation of a combinatorical upper bound (and therfore
        it can be both smaller or greater than the actual determinant).

        :param min_eig: smallest eigenvalue to consider
        :param method: which approximation to use; options are
                       `SumMatrix.APPROX_EIG_METHODS`
        :returns: a vector of the approximate eigenvalues of :math:`K`.
        """
        assert method in self.APPROX_EIG_METHODS

        # TODO style: should convert this into a dictionary switch
        if method == 'greedy_fiedler':
            return self._greedy_fiedler(min_eig)
        elif method == 'simultaneous_diag':
            return self._simultaneous_diag(min_eig)
        elif method == 'greedy_weyl':
            return self._greedy_weyl(min_eig)
        else:
            assert False, '{} not a valid method'.format(method)

    def _greedy_fiedler(self, min_eig):
        Q = len(self.Ks)
        n = self.shape[0]
        eigs = np.zeros((Q, n)) + min_eig
        for i, K in enumerate(self.Ks):
            eigs_K = K.eig(min_eig, exact=False)
            eigs[i, :len(eigs_K)] = eigs_K

        eigsT = eigs.T
        for i in range(Q):
            eigsT = eigsT[np.argsort(eigsT[:, :i].sum(-1))]
            eigsT[:, i][::-1].sort()
            eigs = eigsT.T

        return eigs.sum(axis=0)

    def _simultaneous_diag(self, min_eig):
        # LOWER BOUND
        # useful for kernels which have same eigenvectors
        # Potential idea:
        # Partition kernels into subsets which have the same
        # eigendecomposition (perhaps if the class that generates them
        # is the same). Within a partition, we add eigenvalues
        # with 'simulatneous_diag'. Between partitions, we use
        # 'greedy_fiedler'.
        eigs = np.zeros(self.shape[0])
        for K in self.Ks:
            eigs_K = K.eig(min_eig, exact=False)
            eigs[:len(eigs_K)] += eigs_K

        eigs[eigs == 0] = min_eig
        return eigs

    def _greedy_weyl(self, min_eig):
        # can either greedily max a lower bound
        # or greedily minimize an upper bound

        option = 'minimize_ub' # or 'maximize_lb'
        chooser = np.argmin if option == 'minimize_ub' else np.argmax
        reverse = (lambda x: x) if option == 'minimize_ub' else np.flipud

        n = self.shape[0]
        s = max(int(np.log(n)), 5) # s=n/2 is exact weyl
        eigs = np.zeros(self.shape[0])
        eigs_K = np.zeros_like(eigs)
        for K in self.Ks:
            prev_eigs = reverse(np.copy(eigs))
            eigs_small = K.eig(min_eig, exact=False)
            eigs_K[:] = min_eig
            eigs_K[:len(eigs_small)] = eigs_small
            i = 0
            for k in range(n):
                i, eigs[k] = self._greedy_weyl_incremental(
                    i, s, prev_eigs, eigs_K, chooser, k, n)

        eigs[eigs < min_eig] = min_eig
        return eigs

    def _greedy_weyl_incremental(self, i, s, a, b, chooser, k, n):
        lowest_i = max(i - s, 0)
        highest_i = min(s + (s - (i - lowest_i)), n - 1, k)
        # Weyl says that
        # the k-th largest eigenvalue is less than the sum
        # of the i-th largest of the first matrix and
        # the (k - i)-th largest of the second.
        #
        # In the case of minimizing the upper bound, we want to choose
        # the smallest upper bound, so chooser is np.argmin. Vice-versa
        # for min.
        i_candidates = np.arange(lowest_i, highest_i + 1)
        j_candidates = k - i_candidates
        vals = a[i_candidates] + b[j_candidates]
        best = chooser(vals)
        return i_candidates[best], vals[best]
