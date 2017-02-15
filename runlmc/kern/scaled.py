# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

from paramz.transformations import Logexp

from .stationary_kern import StationaryKern
from ..parameterization.parameterized import Parameterized
from ..parameterization.param import Param
from ..util.docs import inherit_doc

# TODO(test)
@inherit_doc
class Scaled(StationaryKern):
    def __init__(self, k):
        name = 'scaled_' + k.name
        super().__init__(name=name)
        self.k = k
        self.link_parameter(k)
        self.scale = Param('scale', 1, Logexp())

    def from_dist(self, dists):
        return self.scale * self.k.from_dist(dists)

    def to_gpy(self):
        # TODO(cleanup) - will correspond to "variance" or something similar
        raise NotImplementedError

    def kernel_gradient(self, dists):
        grad = [self.scale * g for g in self.k.kernel_gradient(dists)]
        grad.append(self.k.from_dist(dists))
        return grad

    def update_gradient(self, grad):
        self.k.update_gradient(grad[:-1])
        self.scale.gradient = grad[-1]
