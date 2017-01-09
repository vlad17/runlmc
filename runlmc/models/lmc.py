# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import logging

import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg
from paramz.transformations import Logexp

from .multigp import MultiGP
from ..linalg.toeplitz import Toeplitz
from ..linalg.kronecker import Kronecker
from ..linalg.psd_matrix import PSDMatrix
from ..linalg.sum_matrix import SumMatrix
from ..parameterization.param import Param
from ..util.docs import inherit_doc
from ..util.interpolation import interp_cubic

_LOG = logging.getLogger(__name__)

@inherit_doc
class LMC(MultiGP):
    """
    The main class of this package, `LMC` implements linearithmic
    Gaussian Process learning in the multi-output case. [TODO(PAPER)].

    .. Note: Currently, only one-dimensional input is supported.

    Upon construction, this class assumes ownership of its parameters and
    does not account for changes in their values.

    The exact kernel that this approximates is the following:

    .. math::

        K_{\\text{exact}}=\sum_{q=1}^Q\\left(A_qA_q^\\top+
             \\boldsymbol\\kappa_q I\\right)
             \circ [k_q(X_i, X_j)]_{ij\in[D]^2} +
             \\boldsymbol\epsilon I

    :math:`[\cdot]_{ij}` represents a block matrix, with rows and columns
    possibly of different widths. :math:`\circ` is the Hadamard product.
    :math:`\\boldsymbol\\epsilon I` is a Gaussian noise
    addition, iid within each output. The input arrays for our observations
    of each of the different outputs are denoted :math:`X_i` and may be
    variable-length. Each :math:`k_q(X_i,X_j)` is built from a stationary
    Mercer kernel :math:`k_q`, where the :math:`ab`-th entry of the
    rectangular matrix is
    :math:`k_q(\\textbf{x}_a^{(i)}, \\textbf{x}_b^{(j)})` with
    :math:`\\textbf{x}_a^{(i)}` as the
    :math:`a`-th input of the input set :math:`X_i` (and correspondingly for
    :math:`j`).

    The :math:`\\left(A_qA_q^\\top+ \\boldsymbol\\kappa_q I\\right)` terms
    create a kernel which captures some linear correlation between outputs.

    This class uses the SKI approximation, which shares a single grid
    :math:`U` as the input array for all the outputs. Then,
    :math:`K_{\\text{exact}}` is interpolated from the approximation kernel
    :math:`K_{\\text{SKI}}`, as directed in
    *Thoughts on Massively Scalable Gaussian Processes* by Wilson, Dann,
    and Nickisch. This is done with sparse interpolation matrices :math:`W`.

    .. math::

        K_{\\text{exact}}\\approx K_{\\text{SKI}} = W K W^\\top +
            \\boldsymbol\\epsilon I

    Above, :math:`K` is a structured kernel over a grid :math:`U`, derived
    from :math:`A_q, k_q` as before. The grid structure enables us to
    express :math:`K` more succintly, relying on the Kronecker product
    :math:`\\otimes`.

    .. math::

        K=\sum_{q=1}^QA_qA_q^\\top \\otimes k_q(U, U)

    Each :math:`A_q` (only a column vector for now) is a parameter of
    this model, with name `a<q>`, where `<q>` is replaced with a specific
    number.

    TODO(new parameters)
    ranks - currently everything will be rank-1
    mean-function - zero-mean for now

    :param Xs: input observations, should be a list of numpy arrays,
               where the numpy arrays are one dimensional.
    :param Ys: output observations, this must be a list of one-dimensional
               numpy arrays, matching up with the number of rows in `Xs`.
    :param normalize: optional normalization for outputs `Ys`.
                           Prediction will be un-normalized.
    :param kernels: a list of (stationary) kernels which constitute the
                    terms of the LMC sums prior to coregionalization. The
                    :math:`q`-th index here corresponds to :math:`k_q` above.
                    This list's length is :math:`Q`
    :param lo: lexicographically smallest point in inducing point grid used
               (by default, a bit less than the minimum of input)
    :param hi: lexicographically largest point in inducing point grid used
               (by default, a bit more than the maximum of input)
    :param m: number of inducing points to use (by default, the total number
              of input points)
    :param str name:
    :raises: :class:`ValueError` if `Xs` and `Ys` lengths do not match.
    :raises: :class:`ValueError` if normalization if any `Ys` have no variance
                                 or values in `Xs` have multiple identical
                                 values.
    :raises: :class:`ValueError` if no kernels
    """
    def __init__(self, Xs, Ys, normalize=True, kernels=None,
                 lo=None, hi=None, m=None, name='lmc'):
        super().__init__(Xs, Ys, normalize=normalize, name=name)

        if not kernels:
            raise ValueError('Number of kernels should be >0')

        self.kernels = kernels
        for k in self.kernels:
            self.link_parameter(k)

        self.n = sum(map(len, Xs))
        _LOG.info('LMC %s generating inducing grid n = %d',
                  self.name, self.n)
        # Grid corresponds to U
        self.inducing_grid, self.m = self._autogrid(Xs, lo, hi, m)

        # Toeplitz(self.dists) is the pairwise distance matrix of U
        self.dists = self.inducing_grid - self.inducing_grid[0]

        # Corresponds to W; block diagonal matrix.
        self.interpolant = self._interpolant(self.Xs, self.inducing_grid)

        _LOG.info('LMC %s grid (n = %d, m = %d) complete, '
                  'generating first SKI kernel',
                  self.name, self.n, self.m)

        self.coreg_vecs = []
        for i in range(len(self.kernels)):
            coreg_vec = np.random.randn(self.output_dim)
            self.coreg_vecs.append(Param('a{}'.format(i), coreg_vec))
            self.link_parameter(self.coreg_vecs[-1])

        self.coreg_diags = []
        for i in range(len(self.kernels)):
            coreg_diag = np.ones(self.output_dim)
            self.coreg_diags.append(
                Param('kappa{}'.format(i), coreg_diag, Logexp()))
            self.link_parameter(self.coreg_diags[-1])

        # Corresponds to epsilon
        self.noise = Param('noise', np.ones(self.output_dim), Logexp())
        self.link_parameter(self.noise)

        self.ski_kernel = self._generate_ski()
        self.y = np.hstack(self.Ys)

        # Predictive weights
        self.alpha = None
        self.nu = None
        self.native_var = None

        _LOG.info('LMC %s fully initialized', self.name)

    TOL = 1e-4

    class _SKI(PSDMatrix):
        def __init__(self, K_sum, W, noise, Xs):
            lens = [len(X) for X in Xs]
            super().__init__(sum(lens))
            self.K_sum = K_sum
            self.W = W
            self.WT = self.W.transpose().tocsr()
            self.noise = np.repeat(noise, lens)

        def as_numpy(self):
            WKT = self.W.dot(self.K_sum.as_numpy().T)
            return self.W.dot(WKT.T) + np.diag(self.noise)

        def matvec(self, x):
            return self.W.dot(self.K_sum.matvec(self.WT.dot(x))) \
                + x * self.noise

    @staticmethod
    def _autogrid(Xs, lo, hi, m):
        if m is None:
            m = sum(len(X) for X in Xs) // len(Xs)

        if lo is None:
            lo = min(X.min() for X in Xs)

        if hi is None:
            hi = max(X.max() for X in Xs)

        delta = (hi - lo) / m
        lo -= 2 * delta
        hi += 2 * delta
        m += 2

        return np.linspace(lo, hi, m), m

    @staticmethod
    def _interpolant(Xs, inducing_grid): # pylint: disable=too-many-locals
        multiout_grid_sizes = np.arange(len(Xs)) * len(inducing_grid)
        Ws = [interp_cubic(inducing_grid, X) for X in Xs]

        row_lens = [len(X) for X in Xs]
        row_ends = np.add.accumulate(row_lens)
        row_begins = np.roll(row_ends, 1)
        row_begins[0] = 0
        order = row_ends[-1]

        col_lens = [W.nnz for W in Ws]
        col_ends = np.add.accumulate(col_lens)
        col_begins = np.roll(col_ends, 1)
        col_begins[0] = 0
        width = col_ends[-1]

        ind_starts = np.roll(np.add.accumulate([W.indptr[-1] for W in Ws]), 1)
        ind_starts[0] = 0
        ind_ptr = np.append(np.repeat(ind_starts, row_lens), width)
        data = np.empty(width)
        col_indices = np.repeat(multiout_grid_sizes, col_lens)
        for rbegin, rend, cbegin, cend, W in zip(
                row_begins, row_ends, col_begins, col_ends, Ws):
            ind_ptr[rbegin:rend] += W.indptr[:-1]
            data[cbegin:cend] = W.data
            col_indices[cbegin:cend] += W.indices

        ncols = len(Xs) * len(inducing_grid)
        return scipy.sparse.csr_matrix(
            (data, col_indices, ind_ptr), shape=(order, ncols))

    def _generate_ski(self):
        coreg_mats = [np.outer(a, a) + np.diag(k)
                      for a, k in zip(self.coreg_vecs, self.coreg_diags)]
        kernels = [Toeplitz(k.from_dist(self.dists))
                   for k in self.kernels]
        products = [Kronecker(A, K) for A, K in zip(coreg_mats, kernels)]
        kern_sum = SumMatrix(products)
        return LMC._SKI(
            kern_sum,
            self.interpolant,
            self.noise,
            self.Xs)

    def parameters_changed(self):
        # derivatives w.r.t. ordinary covariance hyperparameters
        # d lam(K) = diag(V'*dK*V), for psd matrix K = V*diag(lam)*V'.
        if _LOG.isEnabledFor(logging.DEBUG):
            _LOG.debug('params changed for LMC\n%s', str(self))
        self.ski_kernel = self._generate_ski()

        self.nu = None
        self.alpha = None
        self.native_var = None

    def K_SKI(self):
        """
        .. warning:: This generates the entire kernel, a quadratic operation
                     in memory and time.

        :returns: :math:`K_{\\text{SKI}}`, the approximation of the exact
                  kernel.
        """
        return self.ski_kernel.as_numpy()

    def log_det_K(self):
        """
        :returns: an upper bound of the approximate log determinant,
                  uses :math:`K_\\text{SKI}` to find an approximate
                  upper bound for
                  :math:`\\log\\det K_{\text{exact}}`
        """
        min_noise = min(self.noise.min(), self.TOL)
        eigs = self.ski_kernel.K_sum.approx_eigs(min_noise)
        # noise needs to be adjusted dimensionally. Idea: use top eigs?
        eigs[::-1].sort()
        noise = np.repeat(self.noise, list(map(len, self.Ys)))
        noise.sort()
        # KISS-GP section 2.3.1 from Wilson 2015
        # Theorem 3.4 of Baker 1977 in
        # The numerical treatment of integral equation
        # This converges to the upper bound in the number of examples.
        top_eigs = eigs[:len(noise)] * len(noise) / len(self.inducing_grid)
        return np.log(top_eigs + noise + self.TOL).sum()

    def _invmul(self, y):
        op = self.ski_kernel.as_linear_operator()
        Kinv_y, succ = scipy.sparse.linalg.minres(
            op, y, tol=self.TOL, maxiter=self.m)
        if succ != 0:
            _LOG.critical('MINRES (m = %d) for LMC %s did not converge.\n'
                          'Error %d\nReconstruction %f\nTolerance %f',
                          self.m, self.name, succ,
                          np.linalg.norm(y - op.matvec(Kinv_y)),
                          self.TOL)
        return Kinv_y

    def _precompute_predict(self):
        # Not optimized for speed - this performs dense linear algebra
        # corresponding to Sections 5.1.1 and 5.1.2 from the MSGP paper.
        # This can probably be sped up (even by e.g., using the approximations
        # offered in those sections, or perhaps some less coarse ones).
        # However, it's the training that's the bottleneck, not prediction.
        nongrid_alpha = self._invmul(self.y)
        WT = self.ski_kernel.WT
        K_UU = self.ski_kernel.K_sum
        alpha = K_UU.matvec(WT.dot(nongrid_alpha))

        A = self.K_SKI()
        K_XU = self.interpolant.dot(K_UU.as_numpy())
        Ainv_KXU = la.solve(A, K_XU, sym_pos=True, overwrite_a=True)
        nu = np.diag(K_XU.T.dot(Ainv_KXU))

        # A bit obscure; the native covariance K_** for each output
        # is given by diag(K(0, 0)). This happens to be efficiently computed
        # here.
        coregs = np.square(np.column_stack(self.coreg_vecs))
        coregs += np.column_stack(self.coreg_diags)
        kerns = [k.from_dist(0) for k in self.kernels]
        native_output_var = coregs.dot(kerns).reshape(-1)
        native_var = native_output_var + self.noise

        return alpha, nu, native_var

    def normal_quadratic(self):
        """
        If the flattened (Stacked)outputs are written as :math:`\\textbf{y}`,
        this returns :math:`\\textbf{y}^\\top K_{\\text{SKI}}^{-1}\\textbf{y}`.

        :returns: the normal quadratic term for the current outputs `Ys`.
        """
        return self.y.dot(self._invmul(self.y))

    def log_likelihood(self):
        nll = self.log_det_K() + self.normal_quadratic()
        nll += len(self.y) * np.log(2*np.pi)
        return -0.5 * nll

    def _raw_predict(self, Xs):
        if self.alpha is None:
            self.alpha, self.nu, self.native_var = self._precompute_predict()

        W = self._interpolant(Xs, self.inducing_grid)
        lens = [len(X) for X in Xs]

        mean = W.dot(self.alpha)

        native_var = np.repeat(self.native_var, lens)

        var = native_var - W.dot(self.nu)
        var[var < 0] = 0

        endpoints = np.add.accumulate(lens)[:-1]
        return np.split(mean, endpoints), np.split(var, endpoints)
