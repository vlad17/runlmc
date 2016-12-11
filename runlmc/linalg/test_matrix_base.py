# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import numpy as np

class MatrixTestBase:

    def __init__(self):
        super().__init__()
        # Attributes to be filled in by subclasses

        # Eigenvalue cutoff
        self.eigtol = None

        self.examples = None # List of PSDMatrix being tested

        # Same as above, but for testing approximate eigenvalues
        # matrices here should be well-behaved.
        self.approx_examples = None

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
            x = np.arange(len(np_mat)) + 1
            np.testing.assert_allclose(my_mat.matvec(x), np_mat.dot(x),
                                       err_msg='\n{!s}\n'.format(my_mat))

class DecomposableMatrixTestBase(MatrixTestBase):

    def test_exact_eig(self):
        for my_mat in self.examples:
            np_mat = my_mat.as_numpy()
            np_eigs = np.linalg.eigvalsh(np_mat).real
            np_eigs = np_eigs[np_eigs > self.eigtol]
            np_eigs[::-1].sort()
            np.testing.assert_allclose(my_mat.eig(self.eigtol, exact=True),
                                       np_eigs,
                                       err_msg='\n{!s}\n'.format(my_mat))

    def test_approx_eig(self):
        for my_mat in self.approx_examples:
            np_mat = my_mat.as_numpy()
            sign, logdet = np.linalg.slogdet(np_mat)
            assert sign > 0, sign
            eigs = my_mat.eig(self.eigtol, exact=False)
            my_logdet = np.log(eigs).sum()
            my_logdet += (my_mat.shape[0] - len(eigs)) * np.log(self.eigtol)
            rel_err = abs(logdet - my_logdet)
            rel_err /= 1 if logdet == 0 else abs(logdet)
            msg = '\nmy logdet {} np logdet {}\n{!s}\n'.format(
                my_logdet, logdet, my_mat)
            print(msg)
            self.assertGreaterEqual(0.5, rel_err, msg=msg)

    def test_bound(self):
        for my_mat in self.examples:
            np_mat = my_mat.as_numpy()
            np_eig = np.linalg.eigvalsh(np_mat).real.max()
            ub = my_mat.upper_eig_bound()
            self.assertGreaterEqual(ub, np_eig,
                                    msg='\n{!s}\n'.format(my_mat))
