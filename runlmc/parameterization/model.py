# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

# This file was modified from the GPy project. Its file header is replicated
# below. Its LICENSE.txt is replicated in the LICENSE file for this directory.

# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""
This module defines a generic internal :class:`Model` class, which handles
the interface between this class and the :mod:`paramz` optimization layer.
"""

import numpy as np
import paramz
from paramz.transformations import Transformation, __fixed__

from .priorizable import _PriorizableNode
from ..util.docs import inherit_doc

@inherit_doc
class Model(paramz.Model, _PriorizableNode):
    """
    A :class:`Model` provides a graphical model dependent on latent parameters,
    which it contains either explicitly (as attributes in derived classes
    of :class:`Model`) or implicitly (as parameters implicitly linked to
    a model's explicit parameters).

    Access to any parameter in this tree can be done by the name of those
    parameters. See the :class:`Parameterized` documentation for details.

    The parameters can be either :class:`Param`s or :class:`Parameterized`.
    In fact, the tree of references from objects to their attributes, as
    represented in the Python garbage collector, matches identically with
    the graphical model that this :class:`Model` represents (though the
    direction of the links is reversed).

    The :class:`Model` binds together likelihood values computed from
    the model without the priors (which is implemented by derived classes)
    with the priors. In other words, for observations :math:`\mathbf{y}`,
    parameters :math:`\\theta` dependent on priors :math:`\\phi`, the user
    supplies :math:`\log p(\mathbf{y}|\\theta,\\phi)` as well as its
    derivative with respect to :math:`\\theta`. This class automatically
    adds in the missing :math:`\log p(\\theta|\\phi)` term and its derivative.
    """

    def log_likelihood(self):
        """
        :return: the log likelihood of the current model with respect to its
                 current inputs and outputs and the current prior.
                 This should NOT include the likelihood of the parameters
                 given their priors. In other words, this value should be
                 :math:`\log p(\mathbf{y}|\\theta,\\phi)`
        """
        raise NotImplementedError

    def log_likelihood_with_prior(self):
        """
        Let the observations be :math:`\mathbf{y}`,
        parameters be :math:`\\theta`, and the prior :math:`\\phi`.

        .. math::

            \log p(\mathbf{y}|\\phi) = \log p(\mathbf{y}|\\theta,\\phi) +
            \log p(\mathbf{y}|\\theta,\phi)

        :return: the overall log likelihood shown above.
        """
        return float(self.log_likelihood()) + self.log_prior()

    def objective_function(self):
        return -self.log_likelihood_with_prior()

    def objective_function_gradients(self):
        # self.gradient is the log likelihood without prior gradient
        return -(self.gradient + self._log_prior_gradients())

    def log_prior(self):
        """
        :return: the log prior :math:`\log p(\\theta|\\phi)`
        """
        if self.priors.size == 0:
            return 0.

        log_transformed_prior = sum(
            prior.lnpdf(self.param_array[indices]).sum()
            for prior, indices in self.priors.items())

        # Some of the parameters may have been transformed, so we need
        # to account for their Jacobian factor

        log_jacobian_prior = 0.
        indices_with_prior = {
            idx for _, indices in self.priors.items() for idx in indices}
        for constraint, indices_with_constraint in self.constraints.items():
            if not isinstance(constraint, Transformation):
                continue
            log_jacobian_prior += sum(
                constraint.log_jacobian(self.param_array[i])
                for i in indices_with_constraint
                if i in indices_with_prior)

        return log_transformed_prior + log_jacobian_prior

    def _log_prior_gradients(self):
        if self.priors.size == 0:
            return 0.

        grad = np.zeros(self.param_array.size)
        for prior, indices in self.priors.items():
            np.put(grad, indices, prior.lnpdf_grad(self.param_array[indices]))

        # Incorporate Jacobian for transformations again
        indices_with_prior = {
            idx for idx in indices for _, indices in self.priors.items()}
        for constraint, indices in self.constraints.items():
            if not isinstance(constraint, Transformation):
                continue
            for i in indices:
                if i in indices_with_prior:
                    grad[i] += constraint.log_jacobian_grad(
                        self.param_array[i])

        return grad
