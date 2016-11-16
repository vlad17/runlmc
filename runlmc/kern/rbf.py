# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

# This file was modified from the GPy project. Its file header is replicated
# below. Its LICENSE.txt is replicated in the LICENSE file for this directory.

# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""
The RBF kernel.
"""

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

       k(r) = \sigma^2 \exp \frac{-r^2}{2}
    """
    def __init__(self, variance=1, name='rbf'):
        """
        :param variance: :math:`sigma^2`, above.
        :param name:
        """
        super().__init__(name=name)
        self.variance = Param('variance', variance, Logexp())
        self.link_parameter(self.variance)

    def from_dist(self, dists):
        return self.variance * np.exp(-0.5 * np.square(dists))

    def to_gpy(self):
        import GPy
        v = float(self.variance[0])
        return GPy.kern.RBF(input_dim=1, variance=v, name=self.name)
