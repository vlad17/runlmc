# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

# This file was modified from the GPy project. Its file header is replicated
# below. Its LICENSE.txt is replicated in the LICENSE file for this directory.

# Copyright (c) 2012 - 2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""
This modules contains :class:`Prior`, the base type for all priors available.
"""

import numpy as np
from paramz.domains import _REAL
from scipy.special import gammaln

def _assert_no_constraints(x):
    assert all(c is _REAL for c in x._all_constraints())

class Prior:
    """
    :class:`Prior` allows for incorporating a Bayesian prior in the
    first-order gradient-based optimization performed on the GP models.

    Priors are placed over scalar values.

    Prior objects are immutable.

    Methods are intended to be vectorized over parameters with the same
    priors. In other words, mapping :func:`lnpdf` and
    :func:`lnpdf_grad`
    over each point individually should produce the same result as
    passing in a list of those points.
    """

    _CONSTRAIN_DOMAIN = {
        'real': _assert_no_constraints,
        'positive': lambda x: x.constrain_positive(warning=False),
        'negative': lambda x: x.constrain_negative(warning=False)
    }
    domain = None
    """
    :attribute domain: Domain on which the prior is defined
    :type domain: str
    """

    def lnpdf(self, x):
        """
        :param x: query float or numpy array (for multiple parameters
                  with this same prior)
        :return: the log density of the prior at (each) `x`
        """
        raise NotImplementedError

    def lnpdf_grad(self, x):
        """
        :param x: query float or numpy array (for multiple parameters
                  with this same prior)
        :return: the gradient of the log density of the prior at (each) `x`
        """
        raise NotImplementedError

    @staticmethod
    def _param_print(x):
        from .param import Param
        if isinstance(x, Param):
            return x.name
        return '{}'.format(x)

# TODO(cleanup): test and document below

class Gaussian(Prior):
    domain = 'real'

    def __init__(self, mu, var):
        if var <= 0:
            raise ValueError('variance {} should be positive'.format(var))
        self.mu = mu
        self.var = var
        self.constant = -0.5 * np.log(2 * np.pi * self.var)

    def __str__(self):
        mustr = Prior._param_print(self.mu)
        varstr = Prior._param_print(self.var)
        return "N({}, {})".format(mustr, varstr)

    def lnpdf(self, x):
        return self.constant - 0.5 * np.square(x - self.mu) / self.var

    def lnpdf_grad(self, x):
        return -(x - self.mu) / self.var

class Gamma(Prior):
    domain = 'positive'

    def __init__(self, a, b):
        self.a = float(a)
        self.b = float(b)
        self.constant = -gammaln(self.a) + a * np.log(b)

    def __str__(self):
        astr = Prior._param_print(self.a)
        bstr = Prior._param_print(self.b)
        return "Gamma({},{})".format(astr, bstr)

    def lnpdf(self, x):
        return self.constant + (self.a - 1) * np.log(x) - self.b * x

    def lnpdf_grad(self, x):
        return (self.a - 1.) / x - self.b

    @staticmethod
    def from_EV(E, V):
        """
        Creates an instance of a Gamma Prior  by specifying the Expected value(s)
        and Variance(s) of the distribution.

        :param E: expected value
        :param V: variance
        """
        a = np.square(E) / V
        b = E / V
        return Gamma(a, b)

class InverseGamma(Prior):
    domain = 'positive'

    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.constant = -gammaln(self.a) + a * np.log(b)

    def __str__(self):
        astr = Prior._param_print(self.a)
        bstr = Prior._param_print(self.b)
        return "InverseGamma({},{})".format(astr, bstr)

    def lnpdf(self, x):
        return self.constant - (self.a + 1) * np.log(x) - self.b / x

    def lnpdf_grad(self, x):
        return -(self.a + 1.) / x + self.b / x ** 2

class HalfLaplace:
    domain = 'positive'

    def __init__(self, b):
        self.b = b
        self.constant = - np.log(self.b)

    def __str__(self):
        return "HalfLaplace({})".format(Prior._param_print(self.b))

    def lnpdf(self, x):
        return self.constant - x / self.b

    def lnpdf_grad(self, _):
        return -1 / self.b
