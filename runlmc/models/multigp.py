# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

# This file was modified from the GPy project. Its file header is replicated
# below. Its LICENSE.txt is replicated in the LICENSE file for this directory.

# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""
This module houses the base model that the package centers around:
:class:`runlmc.model.MultiGP`, the parent class for all multi-output
GP models in this package.
"""

import logging

import numpy as np
import scipy.stats
from GPy.util.normalizer import Standardize as Norm

from ..parameterization.model import Model

_LOG = logging.getLogger(__name__)

class MultiGP(Model):
    """
    The generic GP model for multi-output regression. This handles common
    functionality between all models regarding input validation and high
    level parameter optimization routines.

    This model assumes Gaussian noise.

    This class shouldn't be instantiated directly.

    .. Note: Currently, only one-dimensional input is supported.

    Upon construction, this class assumes ownership of its parameters and
    does not account for changes in their values.

    :param Xs: input observations, should be a list of numpy arrays,
               where the numpy arrays are one dimensional.
    :param Ys: output observations, this must be a list of one-dimensional
               numpy arrays, matching up with the number of rows in `Xs`.
    :param normalize: optional normalization for outputs `Ys`.
                           Prediction will be un-normalized.
    :param str name:
    :raises: :class:`ValueError` if `Xs` and `Ys` lengths do not match.
    :raises: :class:`ValueError` if normalization if any `Ys` have no variance
                                 or values in `Xs` have multiple identical
                                 values.
    """
    def __init__(self, Xs, Ys, normalize=True, name='multigp'):
        super().__init__(name)
        self.input_dim, self.output_dim = self._validate_io(Xs, Ys)

        assert self.input_dim == 1, self.input_dim

        self.normalizer = None
        if normalize:
            _LOG.info('%s: normalizing outputs', name)
            self.normalizer = [Norm() for _ in range(len(Ys))]
            for norm, Y in zip(self.normalizer, Ys):
                norm.scale_by(Y)
            Ys = [norm.normalize(Y) for norm, Y in zip(self.normalizer, Ys)]

        self.Xs, self.Ys = Xs, Ys

        _LOG.info('%s: MultiGP initialized', name)

    def parameters_changed(self):
        """
        This method is called automatically when linked parameters change,
        which the may during the optimization process.

        Classes should update their posterior information, log likelihood,
        and gradients when this happens, such that `self._raw_predict`,
        `self.log_likelihood`, and `self.gradient`
        are consistent with the new parameters.

        .. Note: This method should not be called except by the internal
                 usage of `paramz`.
        """
        raise NotImplementedError

    def log_likelihood(self):
        """
        The log marginal likelihood of the model, :math:`p(\mathbf{y})`,
        this is the objective function of the model being optimised
        """
        raise NotImplementedError

    def _raw_predict(self, Xs):
        """
        Returns the raw predictive mean and variance, without
        incorporating normalization.

        See :method:`predict` for more details.
        """
        raise NotImplementedError

    def _predict(self, Xs, normalize):
        assert len(Xs) == self.output_dim, \
            'num inputs {} != output_dim {}'.format(len(Xs), self.output_dim)
        mu, var = self._raw_predict(Xs)

        if self.normalizer and normalize:
            mu = [norm.inverse_mean(m) for norm, m in zip(self.normalizer, mu)]
            var = [norm.inverse_variance(v)
                   for norm, v in zip(self.normalizer, var)]

        return mu, var

    def predict(self, Xs):
        """
        Predict the functions at the new points `Xs`.

        :param Xs: The points at which to make a prediction for each output.
                   Should be empty if no output desired for a certain index.
                   This is a :class:`list` of :class:`numpy.ndarray` for each
                   output `d`, one-dimensional
                   of size `n_d` each. Length of `Xs` should be equal to the
                   number of outputs `self.output_dim`.
        :returns: `(mean, var)`:

            `mean`: posterior mean, a list of length `self.output_dim` with
                    one-dimensional numpy arrays of length `n_d` at index `d`.

            `var`: posterior variance, corresponding to each mean entry.

        .. Note:
            If you want the predictive quantiles (e.g. 95% confidence interval)
            use :func:"predict_quantiles".
        """
        return self._predict(Xs, normalize=True)

    def predict_quantiles(self, Xs, quantiles=(2.5, 97.5)):
        """
        Identical to predict, but returns quantiles instead of mean and
        variance according to the Gaussian likelihood.

        :param quantiles: tuple of quantiles, default is (2.5, 97.5),
                          which is the 95% interval; shouldn't be 0 or 100
        :type quantiles: tuple of doubles
        :returns: list of quantiles for each output's input, as a numpy
                  array, 2-D, the first axis corresponding to the
                  input index and the second to the quantile index.
        """
        mu, var = self._predict(Xs, normalize=False)
        quantiles = np.fromiter(quantiles, dtype=float)
        quantiles = [
            np.outer(np.sqrt(v), scipy.stats.norm.ppf(quantiles/100.))
            + m[:, np.newaxis]
            for m, v in zip(mu, var)]
        if self.normalizer:
            quantiles = [norm.inverse_mean(q)
                         for norm, q in zip(self.normalizer, quantiles)]
        return quantiles

    def optimize(self, **kwargs):
        """
        Optimize the model using :func:`self.log_likelihood` and
        :func:`self.log_likelihood_gradient`, as well as :func:`self.priors`.

        `kwargs` are passed to the optimizer. See parameters for handled
        keywords.

        TODO: do these parameters actually do anything?
        :param max_iters: maximum number of function evaluations
        :type max_iters: int
        :messages: whether to display during optimisation
        :type messages: bool
        :param optimizer: which optimizer to use (defaults to `'lbfgsb'`),
                          options include `'scg'`, `'org-bfgs'`, `'tnc'`,
                          `'adadelta'`, `'rprop'`, `'simplex'`. This can either
                          be a `str` or `paramz.optimization.Optimizer`.
        :param bool ipython_notebook: whether to use ipython notebook widgets
                                      or not (default true)
        :param bool clear_after_finish: if in ipython notebook, we can clear
                                        the widgets after optimization.
                                        Default false.
        """
        try:
            dfldict = {
                'optimizer': None,
                'start': None,
                'messages': False,
                'max_iters': 1000,
                'ipython_notebook': True,
                'clear_after_finish': False
            }
            dfldict.update(kwargs)
            _LOG.info('%s: about to start optimizing - options:\n%s',
                      self.name, dfldict)
            super().optimize(**dfldict)
            _LOG.info('%s: completed optimization')
        except KeyboardInterrupt:
            print('{}: KeyboardInterrupt caught, terminating optimization.'
                  .format(self.name))
            raise

    def _validate_io(self, Xs, Ys):
        if len(Xs) == 0:
            raise ValueError('Expecting at least 1 output')
        if len(Xs) != len(Ys):
            raise ValueError('Differing number of inputs {} and outputs {}'
                             .format(len(Xs), len(Ys)))
        for i, (X, Y) in enumerate(zip(Xs, Ys)):
            if len(X) != len(Y):
                raise ValueError('Ouput {} has {} inputs and {} observed vals'
                                 .format(i, len(X), len(Y)))
            if X.ndim != 1:
                raise ValueError('Input {} mishapen, {} not 1D'
                                 .format(i, X.shape))
            if Y.ndim != 1:
                raise ValueError('Output {} mishapen, {} not 1D'
                                 .format(i, Y.shape))

        for i, Y in enumerate(Ys):
            if np.std(Y) == 0:
                raise ValueError('Output {} has std dev 0'.format(i))

        for i, X in enumerate(Xs):
            u = np.unique(X)
            if len(u) < len(X):
                raise ValueError(
                    'Output {} has {} elements, but only {} unique ones'
                    .format(i, len(X), len(u)))

        return 1, len(Xs)
