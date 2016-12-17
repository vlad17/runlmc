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

    A stationary kernel is a function :math:`k(r)` defined on
    :math:`\\mathbb{R}_+` such that a matrix :math:`K` of entries
    :math:`k(\\|\\textbf{x}_i-\\textbf{x}_j\\|)` is positive semi-definite.
    Moreover, if :math:`\\{\\textbf{x}_i\\}_i` lie on a grid then
    :math:`K` will be block-Toeplitz of Toeplitz blocks. If
    further :math:`\\textbf{x}_i\in\mathbb{R}` then :math:`K` will be
    Toeplitz.

    :param name:
    """
    def __init__(self, name):
        super().__init__(name=name)

    def from_dist(self, dists):
        """
        :param dists: `N`-size numpy array of positive (Euclidean) distances.
        :return: kernel value at each of the given distances
        """
        raise NotImplementedError

    def to_gpy(self):
        """
        :return: GPy version of this kernel.
        """
        raise NotImplementedError

    def kernel_gradient(self, dists):
        """
        Let this kernel be parameterized by some parameters
        :math:`\\boldsymbol\\theta\\in\\mathbb{R}^p`. For every
        :math:`\\theta_j\in\\boldsymbol\\theta`, at any given distance
        :math:`d`, we can compute the derivative
        :math:`\\partial_{\\theta_j}k(d)`. For the evaluation of this
        partial derivative at multiple places, :math:`\\textbf{d}`,
        we call the vector of partial derivatives
        :math:`\\partial_{\\theta_j}k(\\textbf{d})`.

        :param dists: a one-dimensional array of distances corresponding
                      to :math:`\\textbf{d}`, above.
        :returns: An iterable whose :math:`j`-th entry is
                  :math:`\\partial_{\\theta_j}k(\\textbf{d})`.
        """
        raise NotImplementedError

    def update_gradient(self, grad):
        """
        :param grad: a one-dimensional array, representing the gradient vector
                     :math:`\\nabla_{\\boldsymbol\\theta}L` for the
                     likelihood with respect to this kernel's parameters,
                     in the same order of parameters as the row order returned
                     by :func:`kernel_gradient`.
        """
        raise NotImplementedError
