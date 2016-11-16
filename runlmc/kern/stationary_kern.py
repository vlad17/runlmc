# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

"""
This file contains :class:`StationaryKern`, the kernel base class.

Note this class does not accomplish as much as the corresponding
one does in `GPy`. See the class documentation for details.
"""

from ..parameterization.parameterized import Parameterized

class StationaryKern(Parameterized):
    """
    The :class:`StationaryKern` defines a stationary kernel.

    It is a light wrapper around the mathematical definition
    of a kernel, which includes its parameters. A kernel
    object never contains any data, as a parameterized object
    its gradients can be changed according to whatever
    data it's being tuned to.
    """
    def __init__(self, name):
        """
        :param name:
        """
        super().__init__(name=name)

    def from_dist(self, dists):
        """
        :param dists: `N`-size numpy array of positive distances.
        :return: kernel value at each of the given distances
        """
        raise NotImplementedError

    # TODO: something that will give dK_dtheta_k
    # top-level will iterate over each stationary's theta_k,
    # computing dL_dtheta_k efficiently and just setting that index.

    # TODO: make the above a common interface (report deriv of self
    # wrt param, then get told your own derivative). comment this in
    # the README. do this for mean_function as well.

    # TODO: mean_function tests, gradient tests for kern,
    # then see multigp file.

    def to_gpy(self):
        """
        :return: GPy version of this kernel.
        """
        raise NotImplementedError
