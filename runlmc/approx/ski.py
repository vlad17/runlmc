# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import logging
import math

import numpy as np
import scipy.sparse.linalg

from ..linalg.psd_matrix import PSDMatrix

_LOG = logging.getLogger(__name__)

def repeat_noise(Xs, noise):
    """
    :param Xs: inputs
    :param noise: noise to repeat
    :returns: concatenated noise vector from the per-output noise vector,
              repeated according to the number of inputs.
    """
    lens = [len(X) for X in Xs]
    return np.repeat(noise, lens)

class SKI(PSDMatrix):
    def __init__(self, K_sum, W, noise):
        super().__init__(W.shape[0])
        self.m = W.shape[1]
        self.K_sum = K_sum
        self.W = W
        self.WT = self.W.transpose().tocsr()
        self.noise = noise
        self.op = self.as_linear_operator()

    TOL = 1e-10
    """Target solve() tolerance. Only errors > sqrt(TOL) reported."""

    def as_numpy(self):
        WKT = self.W.dot(self.K_sum.as_numpy().T)
        return self.W.dot(WKT.T) + np.diag(self.noise)

    def matvec(self, x):
        return self.W.dot(self.K_sum.matvec(self.WT.dot(x))) \
            + x * self.noise

    def solve(self, y):
        """
        Solves the linear system :math:`K\\textbf{x}=\\textbf{y}`.

        :param y: :math:`\\textbf{y}`
        :return: :math:`\\textbf{x}`
        """
        # K = self.as_numpy()
        # return scipy.linalg.solve(K, y, sym_pos=True, overwrite_a=True)

        Kinv_y, succ = scipy.sparse.linalg.minres(
            self.op, y, tol=self.TOL, maxiter=(self.m ** 2))
        error = np.linalg.norm(y - self.op.matvec(Kinv_y))
        if error > math.sqrt(self.TOL) or succ != 0:
            _LOG.critical('MINRES (m = %d) did not converge.\n'
                          'LMC %s\n'
                          'iterations = m*m = %d\n'
                          'error code %d\nReconstruction Error %f',
                          self.m, self.name, succ, self.m ** 2, error)

        return Kinv_y
