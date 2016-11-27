# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import numpy as np

class MatrixTestBase:

    def setUp(self):
        super().setUp()
        np.random.seed(1234)
        # List of triplets
        # [(matrix being tested, numpy equivalent, diagnostic info)]
        self.examples = None

    @classmethod
    def _print_matrix(cls, x):
        return str(x)

    @staticmethod
    def _assert_wrap(e, info):
        if not e.args:
            e.args = ''
            e.args += '\n{}'.format(info)
        raise e

    @classmethod
    def _rpsd(cls, n):
        A = np.random.randint(-10, 10, (n, n))
        A = (A + A.T).astype(np.float64)
        A += np.diag(np.fabs(A).sum(axis=1) + 1)
        return A

    def test_matvec(self):
        for my_mat, np_mat, info in self.examples:
            x = np.arange(len(np_mat)) + 1
            try:
                np.testing.assert_allclose(my_mat.matvec(x), np_mat.dot(x))
            except AssertionError as e:
                self._assert_wrap(e, info)

    def test_eig(self):
        for my_mat, np_mat, info in self.examples:
            tol = 1e-3
            np_eigs = np.linalg.eigvalsh(np_mat).real
            np_eigs = np_eigs[np_eigs > tol]
            np_eigs[::-1].sort()
            try:
                np.testing.assert_allclose(my_mat.eig(tol), np_eigs)
            except AssertionError as e:
                self._assert_wrap(e, info)
