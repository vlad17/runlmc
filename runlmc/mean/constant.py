# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import numpy as np

from .mean_function import MeanFunction
from ..util.docs import inherit_doc
from ..parameterization.param import Param

@inherit_doc
class Constant(MeanFunction):
    """
    The constant mapping (constant for each output).

    This mean function is not useful with normalization activated.

    It may be useful if you would like to have no normalization
    _and_ want to impose priors on the mean adjustment.

    :param input_dim:
    :param output_dim:
    :param c0: optional vector of length output_dim for the initial offsets
               that the constant takes on.
    """

    def __init__(self, input_dim, output_dim, c0=None, name='constant'):
        super().__init__(input_dim, output_dim, name)

        if c0 is None:
            c0 = np.zeros(output_dim)

        if c0.shape != (output_dim,):
            raise ValueError('Initial constant mean vector shape {} disagrees '
                             'with output dim {}'.format(c0.shape, output_dim))

        self.c = Param('offset', c0)
        self.link_parameter(self.c)

    def f(self, Xs):
        self._validate_inputs(Xs)
        return [np.ones(len(X)) * c for X, c in zip(Xs, self.c)]

    def mean_gradient(self, Xs):
        return [[np.ones(len(X)) if i == j else np.zeros(len(X))
                 for j, X in enumerate(Xs)]
                for i in range(len(self.c))]

    def update_gradient(self, grad):
        self.gradient = grad
