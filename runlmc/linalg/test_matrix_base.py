# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import numpy as np

class MatrixTestBase:

    def __init__(self):
        super().__init__()

        # Attributes to be filled in by subclasses:

        # list of Matrix being tested
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
        for my_mat in self.examples:
            np_mat = my_mat.as_numpy()
            x = np.arange(np_mat.shape[1]) + 1
            np.testing.assert_allclose(my_mat.matvec(x), np_mat.dot(x),
                                       err_msg='\n{!s}\n'.format(my_mat))

    def test_matmat(self):
        for my_mat in self.examples:
            np_mat = my_mat.as_numpy()
            x = np.arange(np_mat.shape[1] * 2).reshape(-1, 2)
            np.testing.assert_allclose(my_mat.matmat(x), np_mat.dot(x),
                                       err_msg='\n{!s}\n'.format(my_mat))
