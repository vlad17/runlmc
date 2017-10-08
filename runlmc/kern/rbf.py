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
class RBF(StationaryKern):
    """
    This class defines the RBF kernel :math:`k`.

    .. math::

       k(r) = \exp \\frac{-\gamma r^2}{2}

    :param inv_lengthscale: :math:`\gamma`, above.
    :param name:
    :param active_dims: see :class:`runlmc.kern.stationary_kern.StationaryKern`
        for details.
    """

    def __init__(self, inv_lengthscale=1, name='rbf', active_dims=None):
        super().__init__(name=name, active_dims=active_dims)
        self.inv_lengthscale = Param(
            'inv_lengthscale', inv_lengthscale, Logexp())
        self.link_parameter(self.inv_lengthscale)

    def from_dist(self, dists):
        return np.exp(-0.5 * np.square(dists) * self.inv_lengthscale)

    def to_gpy(self):
        import GPy
        l = float(self.inv_lengthscale[0]) ** -0.5
        gpy = GPy.kern.RBF(
            input_dim=1, variance=1, lengthscale=l, name=self.name,
            active_dims=self.active_dims)
        gpy.variance.constrain_fixed(1)
        return gpy

    def kernel_gradient(self, dists):
        sqdists = np.square(dists)
        exp = np.exp(-0.5 * sqdists * self.inv_lengthscale)
        return [exp * -0.5 * sqdists]

    def update_gradient(self, grad):
        self.inv_lengthscale.gradient = grad[0]
