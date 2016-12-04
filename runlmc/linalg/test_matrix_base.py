# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import numpy as np

class MatrixTestBase:

    def __init__(self):
        # Attributes to be filled in by subclasses

        # Eigenvalue cutoff
        self.eigtol = None

        # List of pairs
        # [(matrix being tested, numpy equivalent)]
        self.examples = None

    @staticmethod
    def _rpsd(n):
        A = np.random.randint(-10, 10, (n, n))
        A = (A + A.T).astype(np.float64)
        A += np.diag(np.fabs(A).sum(axis=1) + 1)
        return A

    @staticmethod
    def _toep_eig(e, mult):
        # return a psd toeplitz matrix with eigenvalues
        # e (multiplicity mult) and (mult + 1) - mult * e
        assert e > 0
        assert e < 1
        out = np.ones(mult + 1) * 1-e
        out[0] = 1
        return out

    def test_matvec(self):
        for my_mat, np_mat in self.examples:
            x = np.arange(len(np_mat)) + 1
            np.testing.assert_allclose(my_mat.matvec(x), np_mat.dot(x),
                                       err_msg='\n{!s}\n'.format(my_mat))

class DecomposableMatrixTestBase(MatrixTestBase):

    def test_eig(self):
        for my_mat, np_mat in self.examples:
            np_eigs = np.linalg.eigvalsh(np_mat).real
            np_eigs = np_eigs[np_eigs > self.eigtol]
            np_eigs[::-1].sort()
            np.testing.assert_allclose(my_mat.eig(self.eigtol), np_eigs,
                                       err_msg='\n{!s}\n'.format(my_mat))

    def test_bound(self):
        for my_mat, np_mat in self.examples:
            np_eig = np.linalg.eigvalsh(np_mat).real.max()
            ub = my_mat.upper_eig_bound()
            self.assertGreaterEqual(ub, np_eig,
                                    msg='\n{!s}\n'.format(my_mat))
