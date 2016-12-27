# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import unittest

import numpy as np

from .lmc import LMC
from ..util.testing_utils import check_np_lists

class LMCTest(unittest.TestCase):


    def test_empty(self):
        self.assertRaises(ValueError, DummyMultiGP,
                          [], [], False, '')
