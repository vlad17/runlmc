# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import functools
import logging
from multiprocessing import Pool, cpu_count
import os

import numpy as np
import scipy.linalg as la

from .multigp import MultiGP
from ..approx.interpolation import multi_interpolant, autogrid
from ..approx.iterative import Iterative
from ..linalg.composition import Composition
from ..linalg.matrix import Matrix
from ..lmc.stochastic_deriv import StochasticDerivService
from ..lmc.grid_kernel import gen_grid_kernel
from ..lmc.metrics import Metrics
from ..lmc.likelihood import ExactLMCLikelihood, ApproxLMCLikelihood
from ..util.docs import inherit_doc
from ..util.numpy_convenience import cartesian_product
from ..util.inline_pool import InlinePool

_LOG = logging.getLogger(__name__)


@inherit_doc  # pylint: disable=too-many-instance-attributes
class InterpolatedLLGP(MultiGP):
    """
    The main class of this package, `InterpolatedLLGP` implements linearithmic
    Gaussian Process learning in the multi-output case. See
    the paper on `arxiv <https://arxiv.org/abs/1705.10813>`_.

    .. Note: Currently, only one-dimensional input is supported.

    Upon construction, this class assumes ownership of its parameters and
    does not account for changes in their values.

    For a dataset of inputs `Xs` across multiple outputs `Ys`, let :math:`X`
    refer to the concatenation of `Xs`. According to the functional
    specification of the LMC kernel by `functional_kernel` (see documentation
    in :class:`runlmc.lmc.functional_kernel.FunctionalKernel`), we can create
    the covariance matrix for a multi-output GP model applied to all pairs
    of :math:`X`, resulting in :math:`K_{X,X}`.

    The point of this class is to vary hyperparameters of :math:`K`, the
    `FunctionalKernel` given by `functional_kernel`, until the model log
    likelihood is as large as possible.

    This class uses the SKI approximation to do this efficiently,
    which shares a single grid
    :math:`U` as the input array for all the outputs. Then,
    :math:`K_{X,X}` is interpolated from the approximation kernel
    :math:`K_{\\text{SKI}}`, as directed in
    *Thoughts on Massively Scalable Gaussian Processes* by Wilson, Dann,
    and Nickisch. This is done with sparse interpolation matrices :math:`W`.

    .. math::

        K_{X,X}\\approx K_{\\text{SKI}} = W K_{U,U} W^\\top +
            \\boldsymbol\\epsilon I

    Above, :math:`K_{U,U}` is a structured kernel over a grid :math:`U`. This
    grid is specified by `lo,hi,m`.

    The functionality for the various prediction modes is summarized below.

    * `'on-the-fly'` - Use matrix-free inversion to compute the covariance \
    for the entire set of points on which we're predicting. This means that \
    variance prediction take :math:`O(n \log n)` time per test point, where \
    `Xs` has :math:`n` datapoints total. This should be preferred for small \
    test sets.
    * `'precompute'` - Compute an auxiliary predictive variance matrix for \
    the grid points, but then cheaply re-use that work for prediction. This \
    is an up-front :math:`O(n^2 \log n)` payment for :math:`O(1)` predictive \
    variance afterwards per test point.
    * `'exact'` - Use the exact cholesky-based algorithm (not matrix free), \
    :math:`O(n^3)` runtime up-front and then :math:`O(n^2)` per query.

    Note `'on-the-fly', 'precompute'` can be parallelized by the number
    of test points and training points, respectively.

    :param Xs: input observations, should be a list of numpy arrays,
               where each numpy array is a design matrix for the inputs to
               output :math:`i`. If the :math:`i`-th input has :math:`n_i`
               data points, then this matrix can be :math:`n_i` or
               :math:`n_i\\times P` shape for input dimension :math:`P`,
               with the former re-interpreted as :math:`P=1`.
    :param Ys: output observations, this must be a list of one-dimensional
               numpy arrays, matching up with the number of rows in `Xs`.
    :param normalize: optional normalization for outputs `Ys`.
                           Prediction will be un-normalized.
    :param lo: lexicographically smallest point in inducing point grid used
               (by default, a bit less than the minimum of input). For
               multidimensional inputs this should be a vector.
    :param hi: lexicographically largest point in inducing point grid used
               (by default, a bit more than the maximum of input). For
               multidimensional inputs this should be a vector.
    :param m: number of inducing points to use. For multidimensional inputs
              this should be
              a vector indicating how many grid points there should be
              along each dimension. The total number of points used is then
              `np.prod(m)`. By default, `m` is a constant array of dimension
              :math:`P`, the input dimension, of size equal to the average
              input sequence length.
    :param str name:
    :param metrics: whether to record optimization metrics during optimization
                    (runs exact solution alongside this one, may be slow).
    :param prediction: one of `'matrix-free'`, `'on-the-fly'`,
                              `'precompute'`, `'exact'`, `'sample'`.
    :param max_procs: maximum number of processes to use for parallelism,
                      defaults to cpu count.
    :param functional_kernel: a
        :class:`runlmc.lmc.functional_kernel.FunctionalKernel` determining
        :math:`K`.
    :param trace_iterations: number of iterations to be used in approximate
        trace algorithm.
    :raises: :class:`ValueError` if `Xs` and `Ys` lengths do not match.
    :raises: :class:`ValueError` if normalization if any `Ys` have no variance
                                 or values in `Xs` have multiple identical
                                 values.
    :ivar metrics: the :class:`runlmc.lmc.metrics.Metrics` instance associated
                   with the model
    """

    def __init__(self, Xs, Ys, normalize=True,  # pylint: disable=too-many-arguments,dangerous-default-value,too-many-locals
                 lo=None, hi=None, m=None, name='lmc',
                 metrics=False, prediction='on-the-fly', max_procs=None,
                 trace_iterations=10,
                 functional_kernel=None):
        super().__init__(Xs, Ys, normalize=normalize, name=name)
        self.update_model(False)

        if not functional_kernel:
            raise ValueError('functional_kernel must be provided')

        self.prediction = prediction
        if prediction not in self._prediction_methods():
            raise ValueError('Variance prediction method {} unrecognized'
                             .format(prediction))

        n = sum(map(len, self.Xs))
        _LOG.info('InterpolatedLLGP %s generating inducing grid n = %d',
                  self.name, n)
        # Grid corresponds to U
        self.inducing_grid_axes = autogrid(
            self.Xs, self._wrap(lo), self._wrap(hi), self._wrap(m))
        self.grid = cartesian_product(*self.inducing_grid_axes)
        grid_shape = list(map(len, self.inducing_grid_axes))
        grid_shape.append(len(self.inducing_grid_axes))
        self.grid = self.grid.reshape(grid_shape)

        # BTTB(self.dists.ravel(), self.dists.shape)
        # is the pairwise distance matrix of U
        self.dists = la.norm(self.grid - self.grid[0], axis=-1)

        # Corresponds to W; block diagonal matrix.
        self.interpolant = multi_interpolant(self.Xs, *self.inducing_grid_axes)
        self.interpolantT = self.interpolant.transpose().tocsr()

        _LOG.info('InterpolatedLLGP %s grid (n = %d, m = %d) complete, ',
                  self.name, n,
                  np.prod([len(g) for g in self.inducing_grid_axes]))

        self._functional_kernel = functional_kernel
        self.link_parameter(self._functional_kernel)
        self.y = np.hstack(self.Ys)
        self.kernel = None

        self.update_model(True)

        self.metrics = Metrics() if metrics else None
        self._pool = InlinePool(self._generate_pool(max_procs))
        self._deriv_service = StochasticDerivService(
            self.metrics, self._pool, trace_iterations)
        _LOG.info('InterpolatedLLGP %s fully initialized', self.name)

    EVAL_NORM = np.inf

    def _generate_pool(self, max_procs):
        self._check_omp(max_procs)
        max_procs = cpu_count() if max_procs is None else max_procs
        if max_procs == 1 or len(self.y) < 1000:
            _LOG.info('InterpolatedLLGP (%d hyperparams) will run serially',
                      len(self.param_array))
            return None
        _LOG.info('InterpolatedLLGP (%d hyperparams) with %d workers',
                  len(self.param_array), max_procs)
        return Pool(processes=max_procs)

    def _check_omp(self, procs_requested):
        omp = os.environ.get('OMP_NUM_THREADS', None)
        procs_info = 'InterpolatedLLGP(max_procs={})'.format(procs_requested)
        if procs_requested is None:
            procs_info += ' [defaults to {}]'.format(cpu_count())
        if omp is None and (procs_requested or cpu_count()) > 1:
            _LOG.warning('Parallelizing at the process level with %s is'
                         ' incompatible with OMP-level parallelism '
                         '(OMP_NUM_THREADS env var is unset, using all '
                         'available cores)', procs_info)

    def optimize(self, **kwargs):
        if self.metrics is not None:
            self.metrics = Metrics()
            self._deriv_service.metrics = self.metrics
        super().optimize(**kwargs)

    def parameters_changed(self):
        self._clear_caches()

        grid_kernel = gen_grid_kernel(
            self._functional_kernel,
            self.dists,
            self.interpolant,
            self.interpolantT,
            list(map(len, self.Ys)))
        self.kernel = ApproxLMCLikelihood(
            self._functional_kernel,
            grid_kernel,
            self.dists,
            self.Ys,
            self._deriv_service)

        if _LOG.isEnabledFor(logging.DEBUG):
            fmt = '{:7.6e}'.format

            def np_print(x):
                return np.array2string(np.copy(x), formatter={'float': fmt})
            _LOG.debug('Parameters changed')
            _LOG.debug('log likelihood   %f', self.log_likelihood())
            _LOG.debug('normal quadratic %f', self.normal_quadratic())
            _LOG.debug('log det K        %f', self.log_det_K())
            _LOG.debug('noise %s', np_print(self._functional_kernel.noise))
            _LOG.debug('coreg vecs')
            for i, a in enumerate(self._functional_kernel.coreg_vecs):
                _LOG.debug('  a%d %s', i, np_print(a))
            _LOG.debug('coreg diags')
            for i, a in enumerate(self._functional_kernel.coreg_diags):
                _LOG.debug('  kappa%d %s', i, np_print(a))

        self._functional_kernel.update_gradient(self.kernel)

        if self.metrics is not None:
            approx = self.kernel
            exact = self._dense()
            approx_grad, exact_grad = (
                np.concatenate((
                    np.concatenate(
                        k.coreg_vec_gradients()).reshape(-1),
                    np.concatenate(k.coreg_diags_gradients()),
                    np.concatenate(k.kernel_gradients()),
                    k.noise_gradient()))
                for k in [approx, exact])

            approx_norm = la.norm(approx_grad, self.EVAL_NORM)
            exact_norm = la.norm(exact_grad, self.EVAL_NORM)
            diff_norm = la.norm(approx_grad - exact_grad, self.EVAL_NORM)
            self.metrics.grad_norms.append(approx_norm)
            self.metrics.grad_error.append(diff_norm / exact_norm)
            self.metrics.log_likely.append(self.log_likelihood())

    @functools.lru_cache(maxsize=1)
    def _dense(self):
        return ExactLMCLikelihood(
            self._functional_kernel, self.Xs, self.Ys)

    def K(self):
        """
        .. warning:: This generates the entire kernel, a quadratic operation
                     in memory and time.

        :returns: :math:`K_{\\text{SKI}}`, the approximation of the exact
                  kernel.
        """
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
        nll += len(self.y) * np.log(2 * np.pi)
        return -0.5 * nll

    # Predictive mean for grid points U
    @functools.lru_cache(maxsize=1)
    def _grid_alpha(self):
        # TODO(cleanup) use proper kernel interface for this
        return self.kernel.K.grid_K.matvec(
            self.interpolantT.dot(self.kernel.alpha()))

    # The native covariance diag(K) for each output, i.e.,
    # the a priori variance of a single point for each output.
    @functools.lru_cache(maxsize=1)
    def _native_variance(self):
        coregs = np.column_stack(np.square(per_output).sum(axis=0)
                                 for per_output
                                 in self._functional_kernel.coreg_vecs)
        coregs += np.column_stack(self._functional_kernel.coreg_diags)
        kernels = self._functional_kernel.eval_kernels(0)
        native_output_var = coregs.dot(kernels).reshape(-1)
        native_var = native_output_var + self._functional_kernel.noise
        return native_var

    # TODO(test) prediction testing
    def _prediction_methods(self):
        return {
            'on-the-fly': self._var_predict_on_the_fly,
            'precompute': self._var_predict_precompute,
            'exact': self._var_predict_exact,
        }

    def _raw_predict(self, Xs):

        if self.kernel is None:
            self.parameters_changed()

        grid_alpha = self._grid_alpha()
        native_variance = self._native_variance()

        W = multi_interpolant(Xs, *self.inducing_grid_axes)

        mean = W.dot(grid_alpha)
        print(mean.shape)
        lens = [len(X) for X in Xs]
        native_variance = np.repeat(native_variance, lens)

        explained_variance = self._prediction_methods()[self.prediction](
            W, Xs)
        var = native_variance - explained_variance
        var[var < 0] = 0

        endpoints = np.add.accumulate(lens)[:-1]
        return np.split(mean, endpoints), np.split(var, endpoints)

    def _var_predict_exact(self, _, Xs):
        exact = self._dense()
        K_test_X = ExactLMCLikelihood.kernel_from_indices(
            Xs, self.Xs, self._functional_kernel)
        var_explained = K_test_X.dot(la.cho_solve(exact.L, K_test_X.T))

        return np.diag(var_explained)

    @staticmethod
    def _var_solve(i, K, K_XU, K_UX):
        x = np.zeros(K_XU.shape[1])
        x[i] = 1
        x = K_XU.matvec(x)
        x = Iterative.solve(K, x)
        x = K_UX.matvec(x)
        return x[i]

    @functools.lru_cache(maxsize=1)
    def _precomputed_nu(self):
        W = self.interpolant
        WT = self.interpolantT
        K_UU = self.kernel.K.grid_K
        # SKI should have a half_matrix() method returning these two,
        # in which case we'd just ask self.kernel.K.ski for them.
        K_XU = Composition([Matrix.wrap(W.shape, W.dot), K_UU])
        K_UX = Composition([K_UU, Matrix.wrap(WT.shape, WT.dot)])
        Dm = K_XU.shape[1]
        ls = [(i, self.kernel.K, K_XU, K_UX) for i in range(Dm)]
        nu = self._pool.starmap(InterpolatedLLGP._var_solve, ls)

        return nu

    def _var_predict_precompute(self, prediction_interpolant, _):
        nu = self._precomputed_nu()
        return prediction_interpolant.dot(nu)

    def _var_predict_on_the_fly(self, _, Xs):
        K_test_X = ExactLMCLikelihood.kernel_from_indices(
            Xs, self.Xs, self._functional_kernel)
        ls = [(self.kernel.K, k_star) for k_star in K_test_X]
        inverted = np.array(self._pool.starmap(Iterative.solve, ls)).T

        full_mat = K_test_X.dot(inverted)
        return np.diag(full_mat)

    def _clear_caches(self):
        self._dense.cache_clear()
        self._grid_alpha.cache_clear()
        self._native_variance.cache_clear()
        self._precomputed_nu.cache_clear()

    @staticmethod
    def _wrap(x):
        if x is None:
            return None
        n = np.asarray(x)
        if not n.shape:
            return n.reshape(1)
        return n
