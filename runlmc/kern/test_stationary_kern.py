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
        strk = str(k)
        self.assertEqual(len(strk.split('\n')), 2)
        self.assertTrue('asdf' in strk)
        self.assertTrue('variance' in strk)
        self.assertTrue('+ve' in strk)
        self.assertTrue('1' in strk)
