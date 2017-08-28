# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import logging

import scipy.linalg as la
import scipy.sparse.linalg as sla

_LOG = logging.getLogger(__name__)


class _EarlyTerm(Exception):
    def __init__(self, x):
        super().__init__('')
        self.x = x

# TODO(test) + classdoc


class Iterative:
    TOL = 1e-4
    """Target solve() tolerance. Only errors > TOL reported."""

    @staticmethod
    def solve(K, y, verbose=False, minres=True):
        """
        Solves the linear system :math:`K\\textbf{x}=\\textbf{y}`.

        :param K: a :py:class:`SymmetricMatrix`
        :param y: :math:`\\textbf{y}`
        :param verbose: whether to return number of iterations
        :param minres: uses minres if true, else lcg
        :return: :math:`\\textbf{x}`, number of iterations and error if verbose
        """
        ctr = 0

        def cb(x):
            nonlocal ctr, y
            ctr += 1
            if ctr % 100 == 0:  # early termination
                reconstruction = la.norm(y - op.matvec(x))
                if reconstruction < Iterative.TOL:
                    raise _EarlyTerm(x)

        method = sla.minres if minres else sla.cg
        n = K.shape[0]
        op = K.as_linear_operator()
        M = getattr(K, 'preconditioner', None)

        try:
            Kinv_y, succ = method(
                op, y, tol=1e-10, maxiter=n, M=M, callback=cb)
        except _EarlyTerm as e:
            Kinv_y, succ = e.x, 0
        error = la.norm(y - op.matvec(Kinv_y))
        if error > Iterative.TOL or succ != 0:
            _LOG.critical('MINRES (n = %d) did not converge in n iterations.'
                          ' Reconstruction error %e',
                          n, error)

        if verbose:
            return Kinv_y, ctr, error
        return Kinv_y
