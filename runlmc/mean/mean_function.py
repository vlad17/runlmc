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
        """
        Evaluation of the mean function :math:`f`.

        :return:
        """
        raise NotImplementedError

    def _validate_inputs(self, Xs):
        if len(Xs) != self.output_dim:
            raise ValueError('len(Xs) {} != output dimension {}'.format(
                len(Xs), self.output_dim))
        for i, X in enumerate(Xs):
            if X.ndim != 1:
                raise ValueError(
                    'Inputs for output {} have dim {} != 1'.format(
                        i, X.ndim))

    def mean_gradient(self, Xs):
        """
        Let this mean be parameterized by some parameters
        :math:`\\boldsymbol\\theta\\in\\mathbb{R}^p`. For every
        :math:`\\theta_j\in\\boldsymbol\\theta`, at each input point
        :math:`\\textbf{x}^{(i)}` (for a certain output index :math:`i`),
        we can compute the derivative
        :math:`\\partial_{\\theta_j}f(\\textbf{x}^{(i)})`.
        For the evaluation of this
        partial derivative at multiple places, :math:`\\textbf{X}`,
        we call the list of vectors of partial derivatives
        :math:`\\partial_{\\theta_j}f(\\textbf{X})`
        (a list with one vector per output index).

        :param Xs: inputs to evaluate at
        :returns: A list of parameter gradients; the :math:`j`-th entry of this
                  list is :math:`\\partial_{\\theta_j}f(\\textbf{X})`. Each
                  :math:`\\partial_{\\theta_j}f(\\textbf{X})` in turn is
                  another list with one entry per output :math:`i`; the
                  :math:`i`-th entry is a one-dimensional numpy array
                  with :math:`k`-th entry the derivative of the :math:`j`-th
                  mean parameter at the :math:`k`-th input for the :math:`i`-th
                  output, :math:`\\partial_{\\theta_j}f(\\textbf{x}^{(i)}_k)`.
        """
        raise NotImplementedError

    def update_gradient(self, grad):
        """
        :param grad: a one-dimensional array, representing the gradient vector
                     :math:`\\nabla_{\\boldsymbol\\theta}L` for the
                     likelihood with respect to this mean's parameters,
                     in the same order of parameters as the row order returned
                     by :func:`mean_gradient`.
        """
        raise NotImplementedError
