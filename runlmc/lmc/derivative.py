# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

class Derivative:
    def derivative(self, dKdt):
        return 0.5 * (self.d_normal_quadratic(dKdt) - self.d_logdet_K(dKdt))

    def d_normal_quadratic(self, dKdt):
        raise NotImplementedError

    def d_logdet_K(self, dKdt):
        raise NotImplementedError
