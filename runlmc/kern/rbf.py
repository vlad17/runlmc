# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

# This file was modified from the GPy project. Its file header is replicated
# below. Its LICENSE.txt is replicated in the LICENSE file for this directory.

# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""
The RBF kernel.
"""

import math

import numpy as np
from paramz.transformations import Logexp

from .stationary_kern import StationaryKern
from ..parameterization.param import Param
from ..util.docs import inherit_doc

@inherit_doc
class RBF(StationaryKern):
    """
    This class defines RBF kernel :math:`k`.

    .. math::

       k(r) = \sigma^2 \exp \\frac{-\gamma r^2}{2}

    :param variance: :math:`sigma^2`, above.
    :param inv_lengthscale: :math:`gamma`, above.
    :param name:
    """
    def __init__(self, variance=1, inv_lengthscale=1, name='rbf'):
        super().__init__(name=name)
        self.inv_lengthscale = Param(
            'inv_lengthscale', inv_lengthscale, Logexp())
        self.variance = Param('variance', variance, Logexp())
        self.link_parameter(self.variance)
        self.link_parameter(self.inv_lengthscale)

    def from_dist(self, dists):
        exp = np.exp(-0.5 * np.square(dists) * self.inv_lengthscale)
        return self.variance * exp

    def to_gpy(self):
        import GPy
        v = float(self.variance[0])
        l = float(self.inv_lengthscale[0]) ** -0.5
        return GPy.kern.RBF(
            input_dim=1, variance=v, lengthscale=l, name=self.name)

    def kernel_gradient(self, dists):
        # variance gradient, lengthscale gradient
        sqdists = np.square(dists)
        exp = np.exp(-0.5 * sqdists * self.inv_lengthscale)
        return [exp, exp * self.variance * -0.5 * sqdists]

    def update_gradient(self, grad):
        self.variance.gradient = grad[0]
        self.inv_lengthscale.gradient = grad[1]
