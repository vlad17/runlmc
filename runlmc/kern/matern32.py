# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

# This file was modified from the GPy project. Its file header is replicated
# below. Its LICENSE.txt is replicated in the LICENSE file for this directory.

# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from paramz.transformations import Logexp

from .stationary_kern import StationaryKern
from ..parameterization.param import Param
from ..util.docs import inherit_doc


@inherit_doc
class Matern32(StationaryKern):
    """
    This class defines the Matérn-3/2 kernel :math:`k`.

    .. math::

       k(r) = (1+r\\gamma\\sqrt{3})\exp \\left(-\\gamma r\\sqrt{3}\\right)

    :param inv_lengthscale: :math:`\\gamma`, above.
    :param name:
    :param active_dims: see :class:`runlmc.kern.stationary_kern.StationaryKern`
        for details.
    """

    def __init__(self, inv_lengthscale=1, name='matern32', active_dims=None):
        super().__init__(name=name, active_dims=active_dims)
        self.inv_lengthscale = Param(
            'inv_lengthscale', inv_lengthscale, Logexp())
        self.link_parameter(self.inv_lengthscale)

    def from_dist(self, dists):
        scaled_dist = dists * np.sqrt(3) * self.inv_lengthscale
        return (1 + scaled_dist) * np.exp(-scaled_dist)

    def to_gpy(self):
        import GPy
        l = 1 / float(self.inv_lengthscale[0])
        gpy = GPy.kern.Matern32(
            input_dim=1, variance=1, lengthscale=l, name=self.name,
            active_dims=self.active_dims)
        gpy.variance.constrain_fixed(1)
        return gpy

    def kernel_gradient(self, dists):
        scaled_dist = dists * np.sqrt(3) * self.inv_lengthscale
        d_scaled_dist = dists * np.sqrt(3)
        exp = np.exp(-scaled_dist)
        dexp = exp * -d_scaled_dist
        return [(1 + scaled_dist) * dexp + d_scaled_dist * exp]

    def update_gradient(self, grad):
        self.inv_lengthscale.gradient = grad[0]
