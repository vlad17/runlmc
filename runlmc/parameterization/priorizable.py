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

Both :class:`_PriorizableNode` and :class:`_PriorizableLeaf`
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

class _PriorizableLeaf(_PriorizableNode):
    """
    A :class:`_PriorizableLeaf` contains a prior, and, by virtue of
    being a :class:`_PriorizableNode`, will automatically notify
    parents of a new prior being set.
    """

    def set_prior(self, prior):
        """
        Set the prior for this object to prior.

        :param  prior: prior set for this parameter
        :type prior: :class:`runlmc.parameterization.Prior`
        :param bool warning: whether to warn if another prior was set for this
                             parameter
        """
        repriorized = self._unset_priors()
        assert len(repriorized) == 1, 'More than one prior per leaf'
        self._add_to_index_operations(
            self.priors, repriorized, prior, warning=True)

        constrain = Prior.CONSTRAIN_DOMAIN[prior.domain]
        constrain(self)

        assert all(c is not __fixed__ for c in self._all_constraints()), \
            'Should not be fixed with a prior'

    def _all_constraints(self):
        for con in self.constraints.properties_for(self._raveled_index):
            for i in con:
                yield i

    def _unset_priors(self, *priors):
        return self._remove_from_index_operations(self.priors, priors)

    def unset_priors(self, *priors):
        """
        Un-set all priors given in `*priors` from this parameter handle.

        :param priors: the list of :class:`runlmc.priors.Prior` to unset
        """
        self._unset_priors(*priors)
