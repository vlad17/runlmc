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
    def __init__(self, K, W, WT, noise):
        super().__init__(W.shape[0])
        self.m = W.shape[1]
        self.K = K
        self.W = W
        self.WT = WT
        self.noise = noise
        self.op = self.as_linear_operator()

    TOL = 1e-10
    """Target solve() tolerance. Only errors > sqrt(TOL) reported."""

    def as_numpy(self):
        WKT = self.W.dot(self.K.as_numpy().T)
        return self.W.dot(WKT.T) + np.diag(self.noise)

    def matvec(self, x):
        return self.W.dot(self.K.matvec(self.WT.dot(x))) \
            + x * self.noise

    # TODO: move below to more abstract interface?

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
                          'iterations = m*m = %d\n'
                          'error code %d\nReconstruction Error %f',
                          self.m, self.m ** 2, succ, error)

        return Kinv_y

    def logdet_deriv(self, dKdt):
        # Preconditioning Kernel Matrices, Cutajar 2016
        trace = 0
        n = self.shape[0]
        n_it = 4
        for _ in range(n_it):
            r = np.random.randint(0, 2, n) * 2 - 1
            rinv = self.solve(r)
            dr = dKdt.matvec(r)
            trace += rinv.dot(dr)
        return trace / n_it
