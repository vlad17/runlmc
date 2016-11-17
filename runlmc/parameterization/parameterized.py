# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

# This file was modified from the GPy project. Its file header is replicated
# below. Its LICENSE.txt is replicated in the LICENSE file for this directory.

# Copyright (c) 2014, Max Zwiessele, James Hensman
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""
This module contains the :class:`Parameterized` class.

It only exists for convenience, documentation, and to
indicate which methods of the corresponding :mod:`paramz`
class should be used.
"""

import paramz

from .param import Param
from .priorizable import _PriorizableNode

class Parameterized(paramz.Parameterized, _PriorizableNode):
    """
    A :class:`Parameterized` class is responsible for keeping track of
    which parameters it is dependent on.

    It does NOT manage updates
    for those parameters if they change during some optimization
    process. The :class:`runlmc.parametrization.Model` should take care of
    that.

    Say `m` is a handle to a parameterized class.

    Printing parameters:

        - `print(m)`:           prints a nice summary over all parameters

        - `print(name)`:      prints details for param with name 'name'

        - `print m[regexp]`:   prints details for all the parameters
                               which match (!) regexp

        - `print `m['']`:       prints details for all parameters

    Printed fields:

        **name**:         The name of the param, can be renamed!

        **value**:        Shape or value, if one-valued

        **constraints**:  constraint of the param, curly "{c}" brackets
                          indicate
                          some parameters are constrained by c. See detailed
                          print to get exact constraints.

    Getting and setting parameters::

        m.subparameterized1.subparameterized2.param[:] = 1

    Handling of constraining, fixing and tieing parameters:

        You can constrain parameters by calling the constrain on the param
        itself, e.g::

            m.name[:,1].constrain_positive()
            m.name[0].tie_to(m.name[1])

        Fixing parameters will fix them to the value they are right now. If you
        change
        the parameters value, the param will be fixed to the new value!

        If you want to operate on all parameters use `m['']` to wildcard select
        all parameters
        and concatenate them. Printing `m['']` will result in printing of all
        parameters in detail.
    """

    def link_parameter(self, param, index=None):
        """
        Internal API for indicating that a class depends on a certain
        parameter, and therefore should include it in gradient computations.

        .. Note: Developer API. Users should not need to call this.

        :param param:  the parameter to add.
        :type parameters: `runlmc.parametrization.Param`
        :param [index]: index of where to put parameters
        :raise TypeError: if `param` is not the `runlmc` (priorizable)
                          parameter.
        """
        if not isinstance(param, Param):
            raise TypeError(
                'Only runlmc.parametrization.Param types should be linked')
        super().link_parameter(param, index)

    def visit_gradients(self, visit, inputs):
        """
        Update the gradients for the parameters contained here given an
        arbitrary transformation procedure `visit` at the given locations
        `inputs` simultaneously.

        Given a likelihood :math:`L` that we are trying to globally optimize,
        but this class is not directly aware of, this method lets us
        delegate the derivative computation in a generic manner.

        This :class:`Parameterized` object contributes to :math:`L`
        in a way that's defined by its parameters. Let this contribution
        be :math:`C`, a function of parameters :math:`\{\\theta_j\}`.

        There may be other parameters for other :class:`Parameterized` objects
        contributing to :math:`L`, but they don't affect
        :math:`\partial_{\\theta_j}L`.

        The goal of this method is to compute for each :math:`\\theta_j`:

        .. math::

            \partial_{\\theta_j}L(X)=\partial_{C_X}L
                \cdot\partial_{\\theta_j}C_X

        Above, :math:`X` may be a vector of (possibly multi-dimensional)
        inputs, and :math:`C` may be vector-valued function of :math:`\\theta_j`
        for other parameters :math:`\\theta_{-j}` fixed and for a single input
        (element in :math:`X`). Then the derivatives above can be matrices,
        in which case the dot product is generalized.

        To support algorithms that avoid computing the large, dense matrix
        :math:`\partial_{C_X}L`, we allow for a `visit` function :math:`v`
        that should satisfy:

        .. math::

            v(\partial_{\\theta_j}C_X)=\partial_{\\theta_j}L(X)

        Note :math:`\partial_{\\theta_j}C_X` is just as large as
        :math:`\partial_{C_X}L` but the former is sparse.

        :param visit: a function which transforms the local derivative
                      of this class
        """
        raise NotImplementedError
