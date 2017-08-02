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
class GPyLMC(MultiGP):
    """
    This wraps GPy for the Gaussian Process model for multioutput regression
    under a Linear Model of Coregionalization.

    This performs the inversion-based cubic-time algorithm.

    .. Note: Because this implementation uses GPy, mean functions and
             normalization are unsupported.

    Uses the Gaussian likelihood. See :class:`runlmc.models.lmc.LMC` for the
    explicit LMC formula.

    The DTCVAR algorithm (the `sparse` parameter) is based on Efficient
    Multioutput Gaussian Processes through Variational Inducing Kernels
    by Álvarez et al. 2010.

    :param Xs: input observations, should be a list of numpy arrays,
               where the numpy arrays are one dimensional.
    :param Ys: output observations, this must be a list of one-dimensional
               numpy arrays, matching up with the number of rows in `Xs`.
    :param kernels: a list of (stationary) kernels which constitute the
                    terms of the LMC sums prior to coregionalization.
    :param ranks: list of ranks for coregionalization factors
    :type ranks: list of integer
    :param name: model name
    :type name: string
    :param sparse: an integer. If 0, uses
                   :py:class:`GPy.models.GPCoregionalizedRegression`,
                   the typical cholesky algorithm.
                   If >0, then this determines the number of inducing points
                   used by the DTCVAR algorithm in
                   use :py:class:`GPy.models.SparseGPCoregionalizedRegression`
    """

    def __init__(self, Xs, Ys, kernels, ranks, name='GPyLMC', sparse=0):
        super().__init__(Xs, Ys, normalize=False, name=name)
        self.gpy_model = GPyLMC._construct_gpy(
            Xs, Ys, kernels, ranks, sparse)

    def _raw_predict(self, Xs):
        pass

    def parameters_changed(self):
        pass

    def log_likelihood(self):
        return self.gpy_model.log_likelihood()

    def optimize(self, **kwargs):
        self.gpy_model.optimize(**kwargs)

    def predict(self, Xs):
        X = np.hstack(Xs)
        lenls = list(map(len, Xs))
        meta = np.repeat(range(len(Xs)), lenls).reshape(-1, 1)
        mu, var = self.gpy_model.predict(
            np.hstack([X.reshape(-1, 1), meta]),
            Y_metadata={'output_index': meta})
        mu = mu.reshape(-1)
        var = var.reshape(-1)
        return tesselate(mu, lenls), tesselate(var, lenls)

    def predict_quantiles(self, Xs, quantiles=(2.5, 97.5)):
        X = np.hstack(Xs)
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
    def _construct_gpy(Xs, Ys, kernels, ranks, sparse):
        import GPy.models as models
        from GPy.util.multioutput import ICM

        kernels = [k.to_gpy() for k in kernels]
        input_dim = 1
        num_outputs = len(Ys)
        Xs = [X.reshape(-1, 1) for X in Xs]
        Ys = [Y.reshape(-1, 1) for Y in Ys]

        K = ICM(input_dim, num_outputs, kernels[0], ranks[0], name='ICM0')
        for kernel, rank, idx in zip(kernels[1:], ranks[1:], count(1)):
            K += ICM(
                input_dim, num_outputs, kernel, rank, name='ICM{}'.format(idx))
        K.name = 'LCM'
        if sparse > 0:
            return models.SparseGPCoregionalizedRegression(
                Xs, Ys, kernel=K, num_inducing=sparse)
        else:
            return models.GPCoregionalizedRegression(Xs, Ys, kernel=K)
