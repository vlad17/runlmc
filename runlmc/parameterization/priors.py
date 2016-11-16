# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

# This file was modified from the GPy project. Its file header is replicated
# below. Its LICENSE.txt is replicated in the LICENSE file for this directory.

# Copyright (c) 2012 - 2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""
This modules contains :class:`Prior`, the base type for all priors available.
"""

from paramz.domains import _REAL

def _assert_no_constraints(x):
    assert all(c is _REAL for c in x._all_constraints)

class Prior:
    """
    :class:`Prior` allows for incorporating a Bayesian prior in the
    first-order gradient-based optimization performed on the GP models.

    Priors are placed over scalar values.

    Prior objects are immutable.

    Methods are intended to be vectorized over parameters with the same
    priors. In other words, mapping :method:`lnpdf` and :method:`lnpdf_grad`
    over each point individually should produce the same result as
    passing in a list of those points.
    """
    CONSTRAIN_DOMAIN = {
        'real': _assert_no_constraints,
        'positive': lambda x: x.constrain_positive(warning=False),
        'negative': lambda x: x.constrain_negative(warning=False)
    }
    domain = None # must be set to one of the above strings

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
