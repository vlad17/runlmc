# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

# This file was modified from the GPy project. Its file header is replicated
# below. Its LICENSE.txt is replicated in the LICENSE file for this directory.

# Copyright (c) 2012 - 2014 the GPy Austhors (see AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from itertools import count

import numpy as np

from .multigp import MultiGP
from ..util.docs import inherit_doc
from ..util.numpy_convenience import tesselate

@inherit_doc
class ExactLMC(MultiGP):
    """
    The exact Gaussian Process model for heteroscedastic multioutput regression
    under a Linear Model of Coregionalization.

    This performs the inversion-based cubic-time algorithm.

    .. Note: Because this implementation uses GPy, mean functions and
             normalization are unsupported.

    Uses the Gaussian likelihood. Put formally, this computes the regular
    GP with the non-stationary kernel:

    .. math::

        \sum_{q=1}^Q(W_qW_q^\\top+\\boldsymbol\\kappa_q I)
             \circ [k_q(X_i, X_j)]_{ij\in[D]^2} +
             \\boldsymbol\epsilon I

    :math:`[\cdot]_{ij}` represents a block matrix, with rows and columns
    possibly of different widths. :math:`\circ` is the Hadamard product.
    :math:`\\boldsymbol\\kappa_q` is a scaling vector for per-sub-kernel
    variances (make sure redundant parameters don't exist in
    :math:`k_q`). :math:`\\boldsymbol\\epsilon I` is a Gaussian noise
    addition, iid within each output.

    :param Xs: input observations, should be a list of numpy arrays,
               where the numpy arrays are one dimensional. The arrays
               are denoted :math:`X_i` above and may have different lengths.
    :param Ys: output observations, this must be a list of one-dimensional
               numpy arrays, matching up with the number of rows in `Xs`.
    :param kernels: a list of (stationary) kernels which constitute the
                    terms of the LMC sums prior to coregionalization. The
                    :math:`q`-th index here corresponds to :math:`k_q` above.
                    This list's length is :math:`Q`
    :param ranks: :math:`Q`-length list of ranks of :math:`W_q`.
    :type ranks: list of integer
    :param name: model name
    :type name: string
    """
    def __init__(self, Xs, Ys, kernels, ranks, name='ExactLMC'):
        super().__init__(Xs, Ys, mean_function=None,
                         normalize=False, name=name)
        self.gpy_model = ExactLMC._construct_gpy(
            Xs, Ys, kernels, ranks)

    def _raw_predict(self, Xs):
        pass

    def parameters_changed(self):
        pass

    def _parameters_changed(self):
        # Temporarily exposing this while we have a dummy
        # exact LMC implementation - won't be necessary when the real
        # parameters_changed is no longer a no-op.
        self.gpy_model.parameters_changed()

    def log_likelihood(self):
        return self.gpy_model.log_likelihood()

    def optimize(self, **kwargs):
        self.gpy_model.optimize(**kwargs)

    def predict(self, Xs):
        X = np.vstack(Xs)
        lenls = list(map(len, Xs))
        meta = np.repeat(range(len(Xs)), lenls).reshape(-1, 1)
        mu, var = self.gpy_model.predict(
            np.hstack([X.reshape(-1, 1), meta]),
            Y_metadata={'output_index': meta})
        return tesselate(mu, lenls), tesselate(var, lenls)

    def predict_quantiles(self, Xs, quantiles=(2.5, 97.5)):
        X = np.vstack(Xs)
        meta = np.repeat(range(len(Xs)), list(map(len, Xs))).reshape(-1, 1)
        Qs = np.array(self.gpy_model.predict_quantiles(
            np.hstack([X.reshape(-1, 1), meta]),
            quantiles=quantiles,
            Y_metadata={'output_index': meta}))
        assert Qs.shape[2] == 1
        Qs = Qs.reshape(len(quantiles), -1)
        # Qs is a (quantiles) x (num examples) array
        # We want (num examples) x (qunatiles)
        return tesselate(Qs.T, map(len, Xs))

    @staticmethod
    def _construct_gpy(Xs, Ys, kernels, ranks):
        kernels = [k.to_gpy() for k in kernels]
        input_dim = 1
        num_outputs = len(Ys)
        Xs = [X.reshape(-1, 1) for X in Xs]
        Ys = [Y.reshape(-1, 1) for Y in Ys]

        from GPy.util.multioutput import ICM
        from GPy.models import GPCoregionalizedRegression
        K = ICM(input_dim, num_outputs, kernels[0], ranks[0], name='ICM0')
        for kernel, rank, idx in zip(kernels[1:], ranks[1:], count(1)):
            K += ICM(
                input_dim, num_outputs, kernel, rank, name='ICM{}'.format(idx))
        K.name = 'LCM'
        return GPCoregionalizedRegression(Xs, Ys, kernel=K)
