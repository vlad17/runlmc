# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import unittest

from .docs import inherit_doc

class A: # pylint: disable=too-few-public-methods
    """A"""
    def __init__(self):
        """A.__init__"""
        pass

    def a(self):
        """A.a"""
        pass

class B: # pylint: disable=invalid-name
    """B"""
    def a(self):
        """B.a"""
        pass
    def b(self):
        """B.b"""
        pass
    def nodoc(self):
        pass

@inherit_doc # pylint: disable=invalid-name,too-few-public-methods
class C(A):
    def __init__(self):
        super().__init__()
    def a(self):
        pass

@inherit_doc # pylint: disable=invalid-name
class D(A, B):
    def a(self):
        pass
    def b(self):
        pass
    def nodoc(self):
        pass

class TestDocs(unittest.TestCase):
    def test_doc_inheritance_public(self):
        self.assertEqual(C.a.__doc__, A.a.__doc__)

    def test_doc_inheritance_private(self):
        self.assertEqual(C.__init__.__doc__, A.__init__.__doc__)

    def test_doc_multi_inheritance(self):
        self.assertEqual(D.a.__doc__, A.a.__doc__)

    def test_nodoc(self):
        self.assertEqual(D.nodoc.__doc__, None)
