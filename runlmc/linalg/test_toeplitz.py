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

        down = lambda x: (np.arange(x) + 1)[::-1]

        self.eigtol = 1e-6
        self.examples = [Toeplitz(np.array(x)) for x in [
            [1],
            [1, 0],
            [1, 1],
            [0, 0],
            [1, -1],
            [3.5] + [0.999] * 5 + [0] * 110,
            self._toep_eig(self.eigtol / 2, 5),
            self._toep_eig(self.eigtol, 5),
            self._toep_eig(self.eigtol * 2, 5),
            down(10),
            utils.exp_decr_toep(30)]]

        self.approx_examples = [Toeplitz(x) for x in [
            utils.exp_decr_toep(10),
            utils.exp_decr_toep(50),
            utils.exp_decr_toep(100)]]

    def test_as_numpy(self):
        for t in self.examples:
            np.testing.assert_array_equal(
                t.as_numpy(), scipy.linalg.toeplitz(t.top))

    def test_bad_shape(self):
        two_d = np.arange(8).reshape(2, 4)
        self.assertRaises(ValueError, Toeplitz, two_d)
        empty = np.array([])
        self.assertRaises(ValueError, Toeplitz, empty)

    def test_bad_type(self):
        cplx = np.arange(5) * 1j
        self.assertRaises(Exception, Toeplitz, cplx)
