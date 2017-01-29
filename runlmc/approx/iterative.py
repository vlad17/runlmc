# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import logging
import math

import numpy as np
import scipy.sparse.linalg as sla

_LOG = logging.getLogger(__name__)

class Iterative:
    TOL = 1e-10
    """Target solve() tolerance. Only errors > sqrt(TOL) reported."""

    @staticmethod
    def solve(K, y, verbose=False, minres=True):
        """
        Solves the linear system :math:`K\\textbf{x}=\\textbf{y}`.

        :param K: a :py:class:`SymmetricMatrix`
        :param y: :math:`\\textbf{y}`
        :param verbose: whether to return number of iterations
        :param minres: uses minres if true, else lcg
        :return: :math:`\\textbf{x}`, number of iterations if verbose
        """
        ctr = 0
        def cb(_):
            nonlocal ctr
            ctr += 1

        method = sla.minres if minres else sla.cg
        n = K.shape[0]
        op = K.as_linear_operator()
        M = getattr(K, 'preconditioner', None)

        Kinv_y, succ = method(
            op, y, tol=Iterative.TOL, maxiter=n, M=M, callback=cb)
        error = np.linalg.norm(y - op.matvec(Kinv_y))
        if error > math.sqrt(Iterative.TOL) or succ != 0:
            _LOG.critical('MINRES (n = %d) did not converge.\n'
                          'iterations = n\n'
                          'error code %d\nReconstruction Error %f',
                          n, succ, error)

        if verbose:
            return Kinv_y, ctr
        else:
            return Kinv_y
