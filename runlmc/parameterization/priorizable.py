# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

# This file was modified from the GPy project. Its file header is replicated
# below. Its LICENSE.txt is replicated in the LICENSE file for this directory.

# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""
.. Note: Developer API

This module contains internal classes having to do with adding
priors and containing parameters that have priors.

Both :class:`_PriorizableNode` and :class:`PriorizableLeaf`
shouldn't be used externally.
"""

from paramz.core.index_operations import ParameterIndexOperations
from paramz.core.parameter_core import Parameterizable
from paramz.transformations import __fixed__

from .priors import Prior


class _PriorizableNode(Parameterizable):
    """
    Mixin which allows derived classes to have linked parameters
    which in turn have priors.

    This class takes care of propogating reporting for parameters
    to contain priors.
    """

    def __init__(self, name, *a, **kw):
        super().__init__(name=name, *a, **kw)
        self.add_index_operation('priors', ParameterIndexOperations())


class PriorizableLeaf(_PriorizableNode):
    """
    A :class:`PriorizableLeaf` contains a prior, and, by virtue of
    being a :class:`_PriorizableNode`, will automatically notify
    parents of a new prior being set.
    """

    def set_prior(self, prior):
        """
        Set the prior for this object to prior.

        :param  prior: prior set for this parameter
        :type prior: :class:`runlmc.parameterization.Prior`
        """
        # Warn if overwriting previous prior
        prev_priors = self._unset_prior()
        self._add_to_index_operations(
            self.priors, prev_priors, prior, warning=True)

        constrain = Prior._CONSTRAIN_DOMAIN[prior.domain]
        constrain(self)

        assert all(c is not __fixed__ for c in self._all_constraints()), \
            'Should not be fixed with a prior'

    def _all_constraints(self):
        for con in self.constraints.properties_for(self._raveled_index):
            for i in con:
                yield i

    def _unset_prior(self):
        return self._remove_from_index_operations(self.priors, [])

    def unset_prior(self):
        """
        Unset prior, if present.
        """
        self._unset_prior()
