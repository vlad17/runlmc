# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

from contextlib import closing
import logging
from multiprocessing import Pool, cpu_count

import numpy as np
import scipy.linalg as la
import scipy.spatial.distance as dist
from paramz.transformations import Logexp

from .multigp import MultiGP
from ..approx.interpolation import multi_interpolant
from ..approx.iterative import Iterative
from ..linalg.composition import Composition
from ..linalg.matrix import Matrix
from ..lmc.parameter_values import ParameterValues
from ..lmc.grid_kernel import gen_grid_kernel
from ..lmc.metrics import Metrics
from ..lmc.kernel import ExactLMCKernel, ApproxLMCKernel
from ..parameterization.param import Param
from ..util.docs import inherit_doc

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
    :param ranks: list of integer ranks for coregionalization factors,
                  defaults to 1 everywhere.
    :param lo: lexicographically smallest point in inducing point grid used
               (by default, a bit less than the minimum of input)
    :param hi: lexicographically largest point in inducing point grid used
               (by default, a bit more than the maximum of input)
    :param m: number of inducing points to use (by default, the total number
              of input points)
    :param str name:
    :param metrics: whether to record optimization metrics during optimization
                    (runs exact solution alongside this one, may be slow).
    :raises: :class:`ValueError` if `Xs` and `Ys` lengths do not match.
    :raises: :class:`ValueError` if normalization if any `Ys` have no variance
                                 or values in `Xs` have multiple identical
                                 values.
    :raises: :class:`ValueError` if no kernels
    """
    def __init__(self, Xs, Ys, normalize=True, kernels=None,
                 ranks=None, lo=None, hi=None, m=None, name='lmc',
                 metrics=False):
        super().__init__(Xs, Ys, normalize=normalize, name=name)

        if not kernels:
            raise ValueError('Number of kernels should be >0')

        self.kernels = kernels
        for k in self.kernels:
            self.link_parameter(k)

        n = sum(map(len, self.Xs))
        _LOG.info('LMC %s generating inducing grid n = %d',
                  self.name, n)
        # Grid corresponds to U
        self.inducing_grid, m = self._autogrid(Xs, lo, hi, m)

        # Toeplitz(self.dists) is the pairwise distance matrix of U
        self.dists = self.inducing_grid - self.inducing_grid[0]

        # Corresponds to W; block diagonal matrix.
        self.interpolant = multi_interpolant(self.Xs, self.inducing_grid)
        self.interpolantT = self.interpolant.transpose().tocsr()

        _LOG.info('LMC %s grid (n = %d, m = %d) complete, ',
                  self.name, n, m)

        self.coreg_vecs = []
        for i in range(len(self.kernels)):
            rank = 1 if ranks is None else ranks[i]
            coreg_vec = np.random.randn(rank, self.output_dim)
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

        self.y = np.hstack(self.Ys)

        self.kernel = None
        self._cache = {k: None for k in
                       ('exact_kernel', 'grid_alpha', 'nu', 'native_var')}

        self.metrics = Metrics() if metrics else None

        _LOG.info('LMC %s fully initialized', self.name)

    EVAL_NORM = np.inf

    # TODO(cleanup): move to interpolation as its own method; test it.
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
        m += 4

        return np.linspace(lo, hi, m), m

    def _clear_cache(self, name):
        self._cache[name] = None

    def optimize(self, **kwargs):
        if self.metrics is not None:
            self.metrics = Metrics()
        super().optimize(**kwargs)

    def parameters_changed(self):
        for key in ('exact_kernel', 'grid_alpha', 'nu', 'native_var'):
            self._clear_cache(key)

        params = ParameterValues.generate(self)
        grid_kernel = gen_grid_kernel(
            params, self.dists, self.interpolant, self.interpolantT)
        self.kernel = ApproxLMCKernel(
            params, grid_kernel, self.dists, self.metrics)

        if _LOG.isEnabledFor(logging.DEBUG):
            fmt = '{:7.6e}'.format
            def np_print(x):
                return np.array2string(np.copy(x), formatter={'float':fmt})
            _LOG.debug('Parameters changed')
            _LOG.debug('log likelihood   %f', self.log_likelihood())
            _LOG.debug('normal quadratic %f', self.normal_quadratic())
            _LOG.debug('log det K        %f', self.log_det_K())
            _LOG.debug('noise %s', np_print(self.noise))
            _LOG.debug('coreg vecs')
            for i, a in enumerate(self.coreg_vecs):
                _LOG.debug('  a%d %s', i, np_print(a))
            _LOG.debug('coreg diags')
            for i, a in enumerate(self.coreg_diags):
                _LOG.debug('  kappa%d %s', i, np_print(a))

        for x, dx in zip(self.coreg_vecs, self.kernel.coreg_vec_gradients()):
            x.gradient = dx
        for x, dx in zip(self.coreg_diags, self.kernel.coreg_diag_gradients()):
            x.gradient = dx
        for k, dk in zip(self.kernels, self.kernel.kernel_gradients()):
            k.update_gradient(dk)
        self.noise.gradient = self.kernel.noise_gradient()

        if self.metrics is not None:
            grad_norm = np.linalg.norm(self.gradient, self.EVAL_NORM)
            ordered_grad = np.concatenate((
                np.concatenate(
                    [x.gradient for x in self.coreg_vecs]).reshape(-1),
                np.concatenate([x.gradient for x in self.coreg_diags]),
                np.concatenate([x.gradient for x in self.kernels]),
                self.noise.gradient))
            exact = self._dense()
            exact_grad = np.concatenate((
                np.concatenate(
                    exact.coreg_vec_gradients()).reshape(-1),
                np.concatenate(exact.coreg_diag_gradients()),
                np.concatenate(exact.kernel_gradients()),
                exact.noise_gradient()))

            self.metrics.grad_norms.append(grad_norm)
            self.metrics.grad_error.append(
                np.linalg.norm(ordered_grad - exact_grad, self.EVAL_NORM)
                / np.linalg.norm(exact_grad, self.EVAL_NORM))
            self.metrics.log_likely.append(self.log_likelihood())

    def _dense(self):
        if self._cache['exact_kernel'] is None:
            pdists = dist.pdist(np.hstack(self.Xs).reshape(-1, 1))
            pdists = dist.squareform(pdists)
            self._cache['exact_kernel'] = ExactLMCKernel(
                ParameterValues.generate(self), pdists)
        return self._cache['exact_kernel']

    DENSE_CUTOFF = 10000

    def K(self):
        """
        .. warning:: This generates the entire kernel, a quadratic operation
                     in memory and time.

        :returns: :math:`K_{\\text{SKI}}`, the approximation of the exact
                  kernel.
        """
        n = len(self.y)
        if n > self.DENSE_CUTOFF:
            raise ValueError('Matrix size {} too large for dense computation'
                             .format(n))
        return self._dense().K

    def log_det_K(self):
        """
        :returns: an upper bound of the approximate log determinant,
                  uses :math:`K_\\text{SKI}` to find an approximate
                  upper bound for
                  :math:`\\log\\det K_{\text{exact}}`
        """
        diag = np.diag(self._dense().L[0])
        lgdet = np.log(diag).sum() * 2
        sgn = np.sign(diag).prod()
        if sgn <= 0:
            _LOG.critical('Log determinant nonpos! sgn %f lgdet %f '
                          'returning -inf', sgn, lgdet)
            return -np.inf
        return lgdet

    def normal_quadratic(self):
        """
        If the flattened (Stacked)outputs are written as :math:`\\textbf{y}`,
        this returns :math:`\\textbf{y}^\\top K_{\\text{SKI}}^{-1}\\textbf{y}`.

        :returns: the normal quadratic term for the current outputs `Ys`.
        """
        return self.y.dot(self.kernel.alpha())

    def log_likelihood(self):
        nll = self.log_det_K() + self.normal_quadratic()
        nll += len(self.y) * np.log(2*np.pi)
        return -0.5 * nll

    # TODO(test)
    def _raw_predict(self, Xs):
        if len(self.inducing_grid) * len(self.Xs) > self.DENSE_CUTOFF:
            return self._raw_predict_apprx(Xs)
        return self._raw_predict_exact(Xs)

    def _raw_predict_exact(self, Xs):
        # Use dense prediction until TODO(fast-prediction) done
        # TODO(cleanup): refactor ExactLMCKernel so theat below is easier.
        exact = self._dense()

        test_Xs = np.hstack(Xs).reshape(-1, 1)
        train_Xs = np.hstack(self.Xs).reshape(-1, 1)
        params = ParameterValues.generate(self)
        params.lens_x = [len(X) for X in Xs]
        K_test_X = ExactLMCKernel(params,
                                  dist.cdist(test_Xs, train_Xs),
                                  invert=False, noise=False).K
        mean = K_test_X.dot(exact.alpha()).reshape(-1)
        var_explained = K_test_X.dot(la.cho_solve(exact.L, K_test_X.T))
        params.lens = params.lens_x
        var = ExactLMCKernel(params, dist.cdist(test_Xs, test_Xs),
                             invert=False, noise=True).K
        var = np.diag(var) - np.diag(var_explained)

        endpoints = np.add.accumulate(params.lens)[:-1]
        return np.split(mean, endpoints), np.split(var, endpoints)

    def _precompute_predict(self):
        # Not optimized for speed - this performs linear algebra
        # corresponding to Sections 5.1.1 and 5.1.2 from the MSGP paper.
        # The assumption is that training is the bottleneck, not prediction.
        #
        # We can't use MSGP sampling-based variance technique here since
        # it requires taking the square root of the covariance matrix.
        #
        # This is fine for IMC (q=1) kernels that preserve grid structure,
        # but fails for LMC kernels that don't have the structure.
        nongrid_alpha = self.kernel.alpha()
        W = self.interpolant
        WT = self.interpolantT
        K_UU = self.kernel.K.grid_only()
        grid_alpha = K_UU.matvec(WT.dot(nongrid_alpha))
        # TODO(cleanup) kernel should have an ski() method, SKI should
        # have a half_matrix() method returning these two.
        K_XU = Composition([Matrix.wrap(W.shape, W.dot), K_UU])
        K_UX = Composition([K_UU, Matrix.wrap(WT.shape, WT.dot)])

        par = max(cpu_count(), 1)
        chunks = K_XU.shape[1] // (3 * par)
        ls = [(i, self.kernel.K, K_XU, K_UX)
              for i in range(K_XU.shape[1])]
        print('Using {} processors in parallel to precompute predictive'
              ' variances'.format(par))
        with closing(Pool(processes=par)) as pool:
            nu = pool.starmap(LMC._var_solve, ls, chunks)

        # A bit obscure; the native covariance K_** for each output
        # is given by diag(K(0, 0)). This happens to be efficiently computed
        # here.
        coregs = np.column_stack(np.square(per_output).sum(axis=0)
                                 for per_output in self.coreg_vecs)
        coregs += np.column_stack(self.coreg_diags)
        kerns = [k.from_dist(0) for k in self.kernels]
        native_output_var = coregs.dot(kerns).reshape(-1)
        native_var = native_output_var + self.noise

        self._cache['grid_alpha'] = grid_alpha
        self._cache['native_var'] = native_var
        self._cache['nu'] = nu

    def _raw_predict_apprx(self, Xs):
        if self._cache['grid_alpha'] is None:
            self._precompute_predict()

        W = multi_interpolant(Xs, self.inducing_grid)
        lens = [len(X) for X in Xs]

        mean = W.dot(self._cache['grid_alpha'])

        native_var = np.repeat(self._cache['native_var'], lens)

        var = native_var - W.dot(self._cache['nu'])
        var[var < 0] = 0

        endpoints = np.add.accumulate(lens)[:-1]
        return np.split(mean, endpoints), np.split(var, endpoints)

    @staticmethod
    def _var_solve(i, K, K_XU, K_UX):
        x = np.zeros(K_XU.shape[1])
        x[i] = 1
        x = K_XU.matvec(x)
        x = Iterative.solve(K, x)
        x = K_UX.matvec(x)
        return x[i]
