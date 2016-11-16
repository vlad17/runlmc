# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import unittest

from paramz.transformations import Logexp

from .stationary_kern import StationaryKern
from ..parameterization.param import Param

class StationaryKernTest(unittest.TestCase):

    def test_name(self):
        name = 'somename'
        k = StationaryKern(name)
        self.assertEqual(name, k.name)

    def test_str(self):
        k = StationaryKern('asdf')
        k.link_parameter(Param('variance', 1, Logexp()))
        s = str(k)
        self.assertEqual(len(s.split('\n')), 2)
        self.assertTrue('asdf' in s)
        self.assertTrue('variance' in s)
        self.assertTrue('+ve' in s)
        self.assertTrue('1' in s)
