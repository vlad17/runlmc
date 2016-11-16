# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import unittest

import numpy as np

from .numpy_convenience import map_entries, tesselate

class TestNumpyConvenience(unittest.TestCase):
    @staticmethod
    def f(x):
        return x * x

    def tesselate_compare(self, flat, lens, expected):
        actual = tesselate(np.array(flat), lens)
        np.testing.assert_equal(actual, list(map(np.array, expected)))

    def test_map_entires_empty(self):
        empty = np.array([])
        np.testing.assert_almost_equal(
            map_entries(TestNumpyConvenience.f, empty), empty)

    def test_map_entries_float(self):
        floats = np.arange(1.0 * 3 * 4 * 5).reshape(1, 3, 4, 5)
        np.testing.assert_almost_equal(
            map_entries(TestNumpyConvenience.f, floats),
            np.square(floats))

    def test_map_entries_int(self):
        ints = np.arange(1 * 3 * 4 * 5).reshape(1, 3, 4, 5)
        np.testing.assert_almost_equal(
            map_entries(TestNumpyConvenience.f, ints),
            np.square(ints))

    def test_tesselate_throws(self):
        with self.assertRaises(ValueError):
            tesselate(np.array([]), [1])
        with self.assertRaises(ValueError):
            tesselate(np.array([1, 2, 3]), [1, 1, 1, 1])
        with self.assertRaises(ValueError):
            tesselate(np.array([1, 2, 3]), [1, 0, 3])
        with self.assertRaises(ValueError):
            tesselate(np.array([1, 2, 3]), [1, 1])
        with self.assertRaises(ValueError):
            tesselate(np.array([1, 2, 3]), [2])
        with self.assertRaises(ValueError):
            tesselate(np.array([1, 2, 3]), [])

    def test_tesselate_empty(self):
        self.tesselate_compare([], [0], [[]])
        self.tesselate_compare([], [0, 0, 0], [[], [], []])
        self.tesselate_compare([1], [0, 1], [[], [1]])

    def test_tesselate_basic(self):
        self.tesselate_compare([1, 2, 3], [1, 2], [[1], [2, 3]])
