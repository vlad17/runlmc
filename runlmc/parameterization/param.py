# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

# This file was modified from the GPy project. Its file header is replicated
# below. Its LICENSE.txt is replicated in the LICENSE file for this directory.

# Copyright (c) 2014, Max Zwiessele
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""
This module contains the :class:`Param` class, used to keep track of
optimization parameters.

Developers familiar with :py:mod:`paramz` can add their own parameters
for custom kernels and priors.

This class differs from the corresponding :mod:`paramz` one because
it is priorizable.
"""

import paramz

from .priorizable import PriorizableLeaf
from ..util.docs import inherit_doc


@inherit_doc
class Param(paramz.Param, PriorizableLeaf):
    """
    A :class:`Param` should be initialized and used just like a
    :class:`paramz.param.Param`. It contains additional functionality for
    adding priors.
    """
