# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import numpy as np

from .mean_function import MeanFunction
from ..util.docs import inherit_doc

@inherit_doc
class Zero(MeanFunction):
    """
    The zero mapping. Note that leaving the `mean_function` parameter
    as none in all of the models does the same job.
    """
    def __init__(self, input_dim, output_dim, name='zero'):
        super().__init__(input_dim, output_dim, name)

    def f(self, Xs):
        self._validate_inputs(Xs)
        return [np.zeros(len(X)) for X in Xs]

    def mean_gradient(self, Xs):
        # Zero function has no parameters
        return []

    def update_gradient(self, grad):
        pass
