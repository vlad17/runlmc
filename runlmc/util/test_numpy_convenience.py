# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import unittest

import numpy as np

from .numpy_convenience import map_entries

class TestNumpyConvenience(unittest.TestCase):
    @staticmethod
    def f(x):
        return x * x

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
