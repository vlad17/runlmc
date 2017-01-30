# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import numpy as np

from .test_matrix_base import MatrixTestBase
from .diag import Diag
from ..util.testing_utils import RandomTest

class DiagTest(RandomTest, MatrixTestBase):

    def setUp(self):
        super().setUp()

        self.raw_examples = [
            np.ones(1),
            np.ones(2),
            np.arange(2),
            np.arange(3)]
        self.raw_examples += [np.random.rand(x) for x in [3, 10, 15, 20]]

        self.examples = list(map(Diag, self.raw_examples))

    def test_as_numpy(self):
        for s, ex in zip(self.examples, self.raw_examples):
            np.testing.assert_allclose(np.diag(s.as_numpy()), ex)

    def test_empty(self):
        empty = np.array([])
        self.assertRaises(ValueError, Diag, empty)

    def test_shape(self):
        matrix = np.arange(2 * 2).reshape(2, 2)
        self.assertRaises(ValueError, Diag, matrix)
