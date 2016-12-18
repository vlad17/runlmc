# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import unittest

import numpy as np

from .identity import Identity
from ..util.numpy_convenience import map_entries

class IdentityTest(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.cases = np.arange(10.0)
        self.identity_f = lambda r: 1 if r == 0 else 0
        self.rbf_df = lambda r: 0
        self.testK = Identity('wierd-name')

    def f(self, r):
        return 1 if r == 0 else 0

    def test_defaults(self):
        k = Identity()
        self.assertEqual(k.name, 'id')

    def test_values(self):
        actual = self.testK.from_dist(self.cases)
        expected = map_entries(self.f, self.cases)
        np.testing.assert_almost_equal(actual, expected)

    def test_gradients(self):
        actual = self.testK.kernel_gradient(self.cases)
        self.assertEqual(actual, [])
