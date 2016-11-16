# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import unittest

from .stationary_kern import StationaryKern

class StationaryKernTest(unittest.TestCase):

    def test_name(self):
        name = 'somename'
        k = StationaryKern(name)
        self.assertEqual(name, k.name)
