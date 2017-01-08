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
class StdPeriodic(StationaryKern):
    """
    This class defines the standard periodic kernel :math:`k`.

    .. math::

       k(r) = \\exp \\left(\\frac{\gamma}{2}\\sin^2 \\frac{\\pi r}{T}\\right)

    :param inv_lengthscale: :math:`\\gamma`, above.
    :param period: :math:`T`, above.
    :param name:
    """
    def __init__(self, inv_lengthscale=1, period=1, name='std_periodic'):
        super().__init__(name=name)
        self.inv_lengthscale = Param(
            'inv_lengthscale', inv_lengthscale, Logexp())
        self.link_parameter(self.inv_lengthscale)
        self.period = Param(
            'period', period, Logexp())
        self.link_parameter(self.period)

    def from_dist(self, dists):
        if np.log(self.period) < -200:
            return np.nan
        sin = np.sin((np.pi / self.period) * dists)
        return np.exp(-0.5 * np.square(sin) * self.inv_lengthscale)

    def to_gpy(self):
        import GPy
        l = float(self.inv_lengthscale[0]) ** -0.5
        p = float(self.period[0])
        gpy = GPy.kern.StdPeriodic(
            input_dim=1, variance=1, lengthscale=l, period=p, name=self.name)
        gpy.variance.constrain_fixed(1)
        return gpy

    def kernel_gradient(self, dists):
        scaled = np.pi / self.period * dists
        sin = np.sin(scaled)
        dsin = np.cos(scaled) * scaled
        dsin *= -1 / self.period * self.inv_lengthscale
        sqsin = np.square(sin)
        exp = np.exp(-0.5 * sqsin * self.inv_lengthscale)
        return [exp * -0.5 * sqsin, exp * -1 * sin * dsin]

    def update_gradient(self, grad):
        self.inv_lengthscale.gradient = grad[0]
        self.period.gradient = grad[1]
