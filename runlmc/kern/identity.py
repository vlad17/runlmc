# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

"""
The identity kernel (no covariance).
"""

from .stationary_kern import StationaryKern
from ..util.docs import inherit_doc


@inherit_doc
class Identity(StationaryKern):
    """
    This class defines identity kernel :math:`k`.

    .. math::

       k(r) = 1_{r=0}

    :param name:
    :param active_dims: see :class:`runlmc.kern.stationary_kern.StationaryKern`
        for details.
    """

    def __init__(self, name='id', active_dims=None):
        super().__init__(name=name, active_dims=active_dims)

    def from_dist(self, dists):
        return (dists == 0.0).astype(float)

    def to_gpy(self):
        raise NotImplementedError('requires manual impl')

    def kernel_gradient(self, dists):
        return []

    def update_gradient(self, grad):
        pass
