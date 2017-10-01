# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import unittest

import numpy as np
from parameterized import parameterized

from .interpolation import cubic_kernel, interp_cubic, autogrid, interp_bicubic
from .interpolation import multi_interpolant
from ..util.numpy_convenience import cartesian_product
from ..util.testing_utils import error_context, vectorize_inputs


class TestInterpolation(unittest.TestCase):  # pylint: disable=too-many-public-methods

    def test_cubic_out_of_range(self):
        self.assertRaises(ValueError, cubic_kernel, np.arange(0, 3, 0.1))
        self.assertRaises(ValueError, cubic_kernel, np.arange(-3, 0, 0.1))

    def test_cubic_kernel_le1(self):
        # include both endpoints
        x = np.hstack((np.arange(-1, 0, 0.1), np.arange(1, 0, -0.1)))
        absx = np.fabs(x)
        expected = 1.5 * absx ** 3 - 2.5 * absx ** 2 + 1
        np.testing.assert_allclose(cubic_kernel(x), expected)

    def test_cubic_kernel_gr1_le2(self):
        # include both endpoints
        x = np.hstack((np.arange(-2, -1, 0.1), np.arange(2, 1, -0.1)))
        absx = np.fabs(x)
        expected = -0.5 * absx ** 3 + 2.5 * absx ** 2 - 4 * absx + 2
        np.testing.assert_allclose(cubic_kernel(x), expected)

    def test_cubic_kernel_shape(self):
        x = np.arange(-2, 2, 0.5)
        assert x.size == 8
        x = x.reshape(2, 2, 2)
        self.assertEqual(cubic_kernel(x).shape, (2, 2, 2))

    def test_cubic_kernel_empty(self):
        self.assertEqual(cubic_kernel(np.array([])).size, 0)

    def test_interp_cubic_raises_grid_ndim(self):
        grid = np.arange(10).reshape(2, -1)
        sample = np.array([3])
        self.assertRaises(ValueError, interp_cubic, grid, sample)

    def test_interp_cubic_raises_samples_ndim(self):
        grid = np.arange(10)
        sample = np.array([3]).reshape(-1, 1)
        self.assertRaises(ValueError, interp_cubic, grid, sample)

    def test_interp_cubic_raises_grid_size(self):
        grid = np.arange(3)
        sample = np.array([1])
        self.assertRaises(ValueError, interp_cubic, grid, sample)

    def test_interp_cubic_size(self):
        for m, n in [(10, 5), (10, 20), (4, 4)]:
            grid = np.linspace(10, 20, m)
            sample = np.logspace(10, 20, n)
            interp = interp_cubic(grid, sample)
            self.assertEqual(interp.shape, (n, m))

    def test_interp_cubic_sample_lower_bound(self):
        grid = np.arange(10)
        expected_interp = np.zeros((1, 10))
        expected_interp[0, 0] = 1
        for i in [-2, -2.5]:
            sample = np.array([i])
            interp = interp_cubic(grid, sample)
            np.testing.assert_allclose(interp.toarray(), expected_interp)

    def test_interp_cubic_sample_upper_bound(self):
        grid = np.arange(10)
        expected_interp = np.zeros((1, 10))
        expected_interp[0, -1] = 1
        for i in [11, 11.5]:
            sample = np.array([i])
            interp = interp_cubic(grid, sample)
            print(i, interp.toarray())
            np.testing.assert_allclose(interp.toarray(), expected_interp)

    def test_interp_cubic(self):
        grid = np.arange(-0.1, 10.1, 0.1)
        sample = np.arange(10) + 0.5
        M = interp_cubic(grid, sample)
        self.assertEqual(M.shape, (len(sample), len(grid)))

        for f in [np.sin, np.cos, np.square]:
            approxf = M.dot(f(grid))
            exactf = f(sample)
            np.testing.assert_allclose(approxf, exactf, err_msg=str(f))

    def test_interp_bicubic_raises_grid_ndim(self):
        gridx = np.arange(10).reshape(2, -1)
        gridy = np.arange(10)
        sample = np.array([[3, 3]])
        self.assertRaises(ValueError, interp_bicubic, gridx, gridy, sample)
        self.assertRaises(ValueError, interp_bicubic, gridy, gridx, sample)

    def test_interp_bicubic_raises_samples_ndim(self):
        grid = np.arange(10)
        sample = np.array([[3], [3]])
        self.assertRaises(ValueError, interp_bicubic, grid, grid, sample)
        sample = np.array([3, 3])
        self.assertRaises(ValueError, interp_bicubic, grid, grid, sample)

    def test_interp_bicubic_raises_grid_size(self):
        gridx = np.arange(3)
        gridy = np.arange(10)
        sample = np.array([[1, 1]])
        self.assertRaises(ValueError, interp_bicubic, gridx, gridy, sample)
        self.assertRaises(ValueError, interp_bicubic, gridy, gridx, sample)

    def test_interp_bicubic_sample_lower_bound(self):
        gridx = np.arange(10)
        gridy = np.linspace(1, 9, 15)
        expected_interp = np.zeros((1, 150))
        expected_interp[0, 0] = 1
        for i in [-2, -2.5]:
            sample = np.array([[i, i]])
            interp = interp_bicubic(gridx, gridy, sample)
            np.testing.assert_allclose(interp.toarray(), expected_interp)

    def test_interp_bicubic_sample_upper_bound(self):
        gridx = np.arange(10)
        gridy = np.linspace(1, 9, 15)
        expected_interp = np.zeros((1, 150))
        expected_interp[-1, -1] = 1
        for i in [11, 11.5]:
            sample = np.array([[i, i]])
            interp = interp_bicubic(gridx, gridy, sample)
            np.testing.assert_allclose(interp.toarray(), expected_interp)

    def test_interp_bicubicx(self):
        gridx = np.arange(-0.1, 10.1, 0.1)
        gridy = np.arange(0.9, 9.1, 0.1)
        sample = np.arange(1, 8) + 0.5
        sample = cartesian_product(sample, sample)
        M = interp_bicubic(gridx, gridy, sample)
        self.assertEqual(M.shape, (len(sample), len(gridx) * len(gridy)))

        prod = cartesian_product(gridx, gridy)
        fs = [
            lambda x, y: x,
            lambda x, y: y,
            lambda x, y: x + y,
            lambda x, y: np.sin(x * y),
            lambda x, y: np.exp(x - y),
            lambda x, y: np.exp(np.cos(x) * y) - y,
            lambda x, y: x ** 2 - y ** 3 + np.sin(x)]
        for f in fs:
            f = vectorize_inputs(f)
            approxf = M.dot(f(prod))
            exactf = f(sample)
            np.testing.assert_allclose(approxf, exactf, err_msg=str(f))

    def test_multi_interpolant_cubic(self):
        grid = np.arange(-0.1, 10.1, 0.1)
        sample1 = np.arange(10) + 0.5
        sample2 = np.sin(np.arange(6)) + 4
        int1 = interp_cubic(grid, sample1)
        int2 = interp_cubic(grid, sample2)

        exact = np.bmat([
            [int1.toarray(), np.zeros((len(sample1), len(grid)))],
            [np.zeros((len(sample2), len(grid))), int2.toarray()]])
        multi = multi_interpolant([sample1, sample2], grid)
        np.testing.assert_equal(exact, multi.toarray())

    def test_multi_interpolant_bicubic(self):
        gridx = np.arange(-0.1, 10.1, 0.1)
        gridy = np.arange(-2, 11, 0.1)
        sample1 = np.column_stack([np.arange(10) + 0.5, np.ones(10)])
        sample2 = np.column_stack([np.sin(np.arange(6)) + 4, np.arange(6)])
        int1 = interp_bicubic(gridx, gridy, sample1)
        int2 = interp_bicubic(gridx, gridy, sample2)

        m = len(gridx) * len(gridy)
        exact = np.bmat([
            [int1.toarray(), np.zeros((len(sample1), m))],
            [np.zeros((len(sample2), m)), int2.toarray()]])
        multi = multi_interpolant([sample1, sample2], gridx, gridy)
        np.testing.assert_equal(exact, multi.toarray())

    @parameterized.expand([
        ([-1], [13], [10]),
        ([5], [9], [3]),
        (None, None, None),
        (None, [13], None),
        ([-1], None, None),
        ([-1], [13], None)
    ])
    def test_autogrid_lo_hi_m(self, lo, hi, m):
        Xs = [np.arange(10), np.arange(4, 12)]
        Xs = [X.reshape(-1, 1) for X in Xs]
        grid = autogrid(Xs, lo, hi, m)[0]
        with error_context('lo {} hi {} m {}\ngrid {}'
                           .format(lo, hi, m, grid)):
            self.assertGreaterEqual(0, grid[1])
            self.assertLessEqual(11, grid[-2])
            grid_lens = np.unique(np.diff(grid))
            span = grid_lens.max() - grid_lens.min()
            np.testing.assert_allclose(span, 0, atol=1e-6)

    @parameterized.expand([
        ([-1, -5], [13, 15], [10, 12]),
        ([5, 3], [9, 12], [3, 4]),
        (None, None, None),
        (None, [13, 15], None),
        ([-1, -5], None, None),
        ([-1, -5], [13, 15], None)
    ])
    def test_autogrid_lo_hi_m_2d(self, lo, hi, m):
        x1 = np.column_stack([np.arange(10), np.arange(-3, 7)])
        x2 = np.column_stack([np.arange(4, 12), np.arange(2, 10)])
        Xs = [x1, x2]
        gridx, gridy = autogrid(Xs, lo, hi, m)
        with error_context('lo {} hi {} m {}\ngridx {}\ngridy {}'
                           .format(lo, hi, m, gridx, gridy)):
            self.assertGreaterEqual(0, gridx[1])
            self.assertLessEqual(11, gridx[-2])
            self.assertGreaterEqual(-3, gridy[1])
            self.assertLessEqual(10, gridy[-2])
            for grid in [gridx, gridy]:
                grid_lens = np.unique(np.diff(grid))
                span = grid_lens.max() - grid_lens.min()
                np.testing.assert_allclose(span, 0, atol=1e-6)
