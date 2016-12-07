# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import numpy as np
import scipy.linalg

from .test_matrix_base import DecomposableMatrixTestBase
from .toeplitz import Toeplitz
from ..util import testing_utils as utils

class ToeplitzTest(utils.RandomTest, DecomposableMatrixTestBase):

    def setUp(self):
        super().setUp()

        random = np.abs(np.hstack([np.random.rand(30), np.zeros(10)]))
        random[::-1].sort()
        random[0] += np.abs(random[1:]).sum()

        down = lambda x: (np.arange(x) + 1)[::-1]

        self.eigtol = 1e-6
        self.examples = [self._generate(x) for x in [
            [1],
            [1, 0],
            [1, 1],
            [0, 0],
            [1, -1],
            [1] + [0.999] * 5 + [0] * 110,
            self._toep_eig(self.eigtol / 2, 5),
            self._toep_eig(self.eigtol, 5),
            self._toep_eig(self.eigtol * 2, 5),
            down(10),
            random]]

        self.approx_examples = [self._generate(x) for x in [
            utils.exp_decr_toep(10),
            utils.exp_decr_toep(50),
            utils.exp_decr_toep(100)]]

    @staticmethod
    def _generate(x):
        x = np.array(x)
        return Toeplitz(x), scipy.linalg.toeplitz(x)

    def test_bad_shape(self):
        two_d = np.arange(8).reshape(2, 4)
        self.assertRaises(ValueError, Toeplitz, two_d)
        empty = np.array([])
        self.assertRaises(ValueError, Toeplitz, empty)

    def test_bad_type(self):
        cplx = np.arange(5) * 1j
        self.assertRaises(Exception, Toeplitz, cplx)
