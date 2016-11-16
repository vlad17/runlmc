# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

# This file was modified from the GPy project. Its file header is replicated
# below. Its LICENSE.txt is replicated in the LICENSE file for this directory.

# Copyright (c) 2013,2014, GPy authors (see AUTHORS.txt).
# Copyright (c) 2015, James Hensman
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from ..parameterization.parameterized import Parameterized

class MeanFunction(Parameterized):
    """
    Base class for mean functions in multi-output regressions.

    .. Note: mean functions are trained to output normalized values
             if normalization is activated in the models.
    """

    def __init__(self, input_dim, output_dim, name='mapping'):
        super().__init__(name=name)
        self.input_dim = input_dim
        self.output_dim = output_dim

        assert self.input_dim == 1, 'Input dimensions must be 1, for now'

    def f(self, Xs):
        raise NotImplementedError

    # TODO - no more update_gradients
    def update_gradients(self, dL_dF, X):
        raise NotImplementedError

    def _validate_inputs(self, Xs):
        if len(Xs) != self.output_dim:
            raise ValueError('len(Xs) {} != output dimension {}'.format(
                len(Xs), self.output_dim))
        for i, X in enumerate(Xs):
            if X.ndim > 1:
                raise ValueError(
                    'Inputs for output {} have dim {} != 1'.format(
                        i, X.ndim))
