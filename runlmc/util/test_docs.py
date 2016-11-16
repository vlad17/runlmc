# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

# pylint: skip-file
# Skipping the file for simply-named test classes.

import unittest

import numpy as np

from .docs import inherit_doc

class A:
    """A"""
    def __init__(self):
        """A.__init__"""
        pass

    def a():
        """A.a"""
        pass

class B:
    """B"""
    def a():
        """B.a"""
        pass
    def b():
        """B.b"""
        pass
    def nodoc():
        pass

@inherit_doc
class C(A):
    def __init__(self):
        super().__init__()
    def a():
        pass

@inherit_doc
class D(A, B):
    def a():
        pass
    def b():
        pass
    def nodoc():
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
