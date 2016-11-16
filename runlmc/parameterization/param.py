# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

# This file was modified from the GPy project. Its file header is replicated
# below. Its LICENSE.txt is replicated in the LICENSE file for this directory.

# Copyright (c) 2014, Max Zwiessele
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""
This module contains the :class:`Param` class, used to keep track of
optimization parameters.

Developers familiar with :module:`paramz` can add their own parameters
for custom kernels and priors.

This class differs from the corresponding :module:`paramz` one because
it is priorizable.
"""

import paramz

from .priorizable import _PriorizableLeaf
from ..util.docs import inherit_doc

@inherit_doc
class Param(paramz.Param, _PriorizableLeaf):
    """
    A :class:`Param` should be initialized and used just like a
    :class:`paramz.Param`. It contains additional functionality for
    adding priors.

    .. Note: Developer API. Most users should not need to interact with this.
    """

    # Repeat base methods we want documented here.

    def __init__(self, name, input_array, default_constraint=None, *a, **kw):
        super().__init__(name, input_array, default_constraint, *a, **kw)
        self.prior = None

    def set_prior(self, prior):
        super().set_prior(prior)

    def unset_priors(self, *priors):
        super().unset_priors(*priors)
