# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import numpy as np
import scipy.linalg as la

from .test_matrix_base import MatrixTestBase
from .bttb import BTTB
from ..util import testing_utils as utils


class BTTBTest(utils.RandomTest, MatrixTestBase):

    def setUp(self):
        super().setUp()

        exs = [
            np.arange(1),
            np.arange(3),
            np.arange(2 * 3).reshape(2, 3),
            np.arange(10),
            np.arange(100),
            np.arange(2 * 3 * 4).reshape(2, 3, 4),
        ]
        exs2 = [np.random.rand(*ex.shape) for ex in exs]
        exs = map(np.array, exs + exs2)
        self.examples = [BTTB(x.ravel(), x.shape) for x in exs]

    def test_as_numpy_1d(self):
        top = np.array([4, 3, 2, 1])
        exact = la.toeplitz(top)
        t = BTTB(top, top.shape)
        np.testing.assert_array_equal(
            t.as_numpy(), exact)

        top = np.array([4])
        exact = la.toeplitz(top)
        t = BTTB(top, top.shape)
        np.testing.assert_array_equal(
            t.as_numpy(), exact)

    def test_as_numpy_2d(self):
        top = np.array([[4, 3, 2, 1], [3, 2, 1, 0], [2, 1, 0, 0]])
        t1, t2, t3 = map(la.toeplitz, top)
        exact = np.bmat(
            [[t1, t2, t3],
             [t2, t1, t2],
             [t3, t2, t1]]).A
        t = BTTB(top.ravel(), top.shape)
        np.testing.assert_array_equal(
            t.as_numpy(), exact)

    def test_as_numpy_3d(self):
        top = np.array([[[4, 3], [2, 1]], [[3, 2], [1, 0]], [[2, 1], [0, 0]]])
        t1, t2, t3, t4, t5, t6 = map(la.toeplitz, top.reshape(-1, 2))
        b1 = np.bmat(
            [[t1, t2],
             [t2, t1]]).A
        b2 = np.bmat(
            [[t3, t4],
             [t4, t3]]).A
        b3 = np.bmat(
            [[t5, t6],
             [t6, t5]]).A
        exact = np.bmat(
            [[b1, b2, b3],
             [b2, b1, b2],
             [b3, b2, b1]]).A
        t = BTTB(top.ravel(), top.shape)
        np.testing.assert_array_equal(
            t.as_numpy(), exact)

    def test_bad_shape(self):
        two_d = np.arange(8).reshape(2, 4)
        self.assertRaises(ValueError, BTTB, two_d, two_d.shape)
        empty = np.array([])
        self.assertRaises(ValueError, BTTB, empty, empty.shape)
        self.assertRaises(ValueError, BTTB, two_d, empty.shape)
        self.assertRaises(ValueError, BTTB, two_d, (3, 4))

    def test_bad_type(self):
        cplx = np.arange(5) * 1j
        self.assertRaises(Exception, BTTB, cplx, cplx.shape)
