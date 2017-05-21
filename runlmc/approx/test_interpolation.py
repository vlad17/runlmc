# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import unittest

import numpy as np

from .interpolation import cubic_kernel, interp_cubic

class TestInterpolation(unittest.TestCase):

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
