# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

from contextlib import closing
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
from ..linalg.numpy_matrix import NumpyMatrix
from ..linalg.kronecker import Kronecker
from ..linalg.diag import Diag
from ..lmc.stochastic_deriv import StochasticDeriv
from ..lmc.grid_kernel import gen_grid_kernel
from ..lmc.metrics import Metrics
from ..lmc.likelihood import ExactLMCLikelihood, ApproxLMCLikelihood
from ..util.docs import inherit_doc

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
    Note `'matrix-free'` is the default.

    * `'matrix-free'` - If the number of test points is smaller than the \
    number of grid points, use `'on-the-fly'`. If the \
    number of points is greater than the number of grid \
    points, use `'precompute'`, and use that from then onwards.
    * `'on-the-fly'` - Use matrix-free inversion to compute the covariance \
    for the entire set of points on which we're predicting.
    * `'precompute'` - Compute an auxiliary predictive variance matrix for \
    the grid points, but then cheaply re-use that work for prediction.
    * `'exact'` - Use the exact cholesky-based algorithm (not matrix free)
    * `'sample'` - Use the sampling algorithm from Wilson 2015.

    :param Xs: input observations, should be a list of numpy arrays,
               where the numpy arrays are one dimensional.
    :param Ys: output observations, this must be a list of one-dimensional
               numpy arrays, matching up with the number of rows in `Xs`.
    :param normalize: optional normalization for outputs `Ys`.
                           Prediction will be un-normalized.
    :param lo: lexicographically smallest point in inducing point grid used
               (by default, a bit less than the minimum of input)
    :param hi: lexicographically largest point in inducing point grid used
               (by default, a bit more than the maximum of input)
    :param m: number of inducing points to use (by default, the total number
              of input points)
    :param str name:
    :param metrics: whether to record optimization metrics during optimization
                    (runs exact solution alongside this one, may be slow).
    :param prediction: one of `'matrix-free'`, `'on-the-fly'`,
                              `'precompute'`, `'exact'`, `'sample'`.
    :param variance_samples: only used if `prediction` is set to `'sample'`.
                             number of variance samples used for predictive
                             variance.
    :param max_procs: maximum number of processes to use for parallelism,
                      defaults to cpu count.
    :param functional_kernel: a
        :class:`runlmc.lmc.functional_kernel.FunctionalKernel` determining
        :math:`K`.
    :raises: :class:`ValueError` if `Xs` and `Ys` lengths do not match.
    :raises: :class:`ValueError` if normalization if any `Ys` have no variance
                                 or values in `Xs` have multiple identical
                                 values.
    """

    def __init__(self, Xs, Ys, normalize=True,  # pylint: disable=too-many-arguments,dangerous-default-value,too-many-locals
                 lo=None, hi=None, m=None, name='lmc',
                 metrics=False, prediction='matrix-free',
                 variance_samples=20, max_procs=None,
                 functional_kernel=None):

        super().__init__(Xs, Ys, normalize=normalize, name=name)
        self.update_model(False)
        # TODO(cleanup) - this entire constructor needs reorg, refactor
        # into smaller methods, etc. Large number of arguments is OK, though.
        # basically address all pylint complaints (disabled in the constructor)

        if not functional_kernel:
            raise ValueError('functional_kernel must be provided')

        self.variance_samples = variance_samples
        self.prediction = prediction
        if prediction not in self._prediction_methods():
            raise ValueError('Variance prediction method {} unrecognized'
                             .format(prediction))

        n = sum(map(len, self.Xs))
        _LOG.info('InterpolatedLLGP %s generating inducing grid n = %d',
                  self.name, n)
        # Grid corresponds to U
        self.inducing_grid, m = autogrid(Xs, lo, hi, m)

        # Toeplitz(self.dists) is the pairwise distance matrix of U
        self.dists = self.inducing_grid - self.inducing_grid[0]

        # Corresponds to W; block diagonal matrix.
        self.interpolant = multi_interpolant(self.Xs, self.inducing_grid)
        self.interpolantT = self.interpolant.transpose().tocsr()

        _LOG.info('InterpolatedLLGP %s grid (n = %d, m = %d) complete, ',
                  self.name, n, m)

        self._functional_kernel = functional_kernel
        self.link_parameter(self._functional_kernel)
        self.y = np.hstack(self.Ys)

        self.kernel = None
        self._cache = {}
        self.metrics = Metrics() if metrics else None
        self.max_procs = cpu_count() if max_procs is None else max_procs
        self._pool = None

        self._check_omp(max_procs)

        self.update_model(True)
        _LOG.info('InterpolatedLLGP %s fully initialized', self.name)

    EVAL_NORM = np.inf

    def _check_omp(self, procs_requested):
        omp = os.environ.get('OMP_NUM_THREADS', None)
        procs_info = 'InterpolatedLLGP(max_procs={})'.format(procs_requested)
        if procs_requested is None:
            procs_info += ' [defaults to {}]'.format(self.max_procs)
        if omp is None and self.max_procs > 1:
            _LOG.warning('Parallelizing at the process level with %s is'
                         ' incompatible with OMP-level parallelism '
                         '(OMP_NUM_THREADS env var is unset, using all '
                         'available cores)', procs_info)

    def optimize(self, **kwargs):
        if self.metrics is not None:
            self.metrics = Metrics()
        if self.max_procs == 1 or len(self.y) < 1000:
            _LOG.info('Optimization (%d hyperparams) starting in serial mode',
                      len(self.param_array))
            self._pool = None
            super().optimize(**kwargs)
        else:
            par = min(StochasticDeriv.N_IT + 1, self.max_procs)
            _LOG.info('Optimization (%d hyperparams) starting with %d workers',
                      len(self.param_array), par)
            with closing(Pool(processes=par)) as pool:
                self._pool = pool
                super().optimize(**kwargs)
            self._pool = None

    def parameters_changed(self):
        self._cache = {}

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
            self.metrics,
            self._pool)

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

    def _dense(self):
        if 'exact_kernel' not in self._cache:
            self._cache['exact_kernel'] = ExactLMCLikelihood(
                self._functional_kernel, self.Xs, self.Ys)
        return self._cache['exact_kernel']

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
    def _grid_alpha(self):
        if 'grid_alpha' not in self._cache:
            self._cache['grid_alpha'] = self.kernel.K.grid_K.matvec(
                self.interpolantT.dot(self.kernel.alpha()))
        return self._cache['grid_alpha']

    # The native covariance diag(K) for each output, i.e.,
    # the a priori variance of a single point for each output.
    def _native_variance(self):
        if 'native_var' not in self._cache:
            coregs = np.column_stack(np.square(per_output).sum(axis=0)
                                     for per_output
                                     in self._functional_kernel.coreg_vecs)
            coregs += np.column_stack(self._functional_kernel.coreg_diags)
            kernels = self._functional_kernel.eval_kernels(0)
            native_output_var = coregs.dot(kernels).reshape(-1)
            native_var = native_output_var + self._functional_kernel.noise

            self._cache['native_var'] = native_var
        return self._cache['native_var']

    # TODO(test) prediction testing
    def _prediction_methods(self):
        return {
            'matrix-free': self._var_predict_matrix_free,
            'on-the-fly': self._var_predict_on_the_fly,
            'precompute': self._var_predict_precompute,
            'exact': self._var_predict_exact,
            'sample': self._var_predict_sample
        }

    def _raw_predict(self, Xs):

        if self.kernel is None:
            self.parameters_changed()

        grid_alpha = self._grid_alpha()
        native_variance = self._native_variance()

        W = multi_interpolant(Xs, self.inducing_grid)

        mean = W.dot(grid_alpha)
        lens = [len(X) for X in Xs]
        native_variance = np.repeat(native_variance, lens)

        explained_variance = self._prediction_methods()[self.prediction](
            W, Xs)
        var = native_variance - explained_variance
        var[var < 0] = 0

        endpoints = np.add.accumulate(lens)[:-1]
        return np.split(mean, endpoints), np.split(var, endpoints)

    # TODO(cleanup) all of the below variance prediction methods should
    # be cleaned up and moved to a separate module.

    def _var_predict_matrix_free(self, W, Xs):
        if 'precomputed_nu' in self._cache:
            return self._var_predict_precompute(W, Xs)

        tot_pred_size = sum(map(len, Xs))
        tot_grid_size = len(self.inducing_grid) * len(
            self._functional_kernel.noise)
        if tot_pred_size > tot_grid_size:
            return self._var_predict_precompute(W, Xs)
        return self._var_predict_on_the_fly(W, Xs)

    def _var_predict_exact(self, _, Xs):
        exact = self._dense()
        K_test_X = ExactLMCLikelihood.kernel_from_indices(
            Xs, self.Xs, self._functional_kernel)
        var_explained = K_test_X.dot(la.cho_solve(exact.L, K_test_X.T))

        return np.diag(var_explained)

    @staticmethod
    def _chol_sample(W, B, t, randn_samps, q):
        LB = la.cholesky(B)
        # bareiss and cholesky both work very poorly since we're
        # numerically rank deficient
        # from runlmc.linalg.shur import shur
        # Lt = shur(t).T
        # Lt = la.cholesky(la.toeplitz(t))
        w, v = la.eigh(la.toeplitz(t))
        w[w < 0] = 0
        nnz = np.count_nonzero(w)
        if nnz < len(w):
            _LOG.info('encountered incomplete rank %d of %d order kernel %d',
                      nnz, len(w), q)
        Lt = v * np.sqrt(w)
        L = Kronecker(NumpyMatrix(LB), NumpyMatrix(Lt))
        return W.dot(L.matmat(randn_samps))

    def _sampled_nu(self):
        if 'sampled_nu' in self._cache:
            return self._cache['sampled_nu']

        # This performs linear algebra
        # corresponding to Sections 5.1.1 and 5.1.2 from the MSGP paper.
        #
        # The MSGP sampling-based variance technique here requires
        # a cholesky decomposition, done in a parallalized manner.
        # Thus, we're not matrix-free in the predictions.

        Ns = self.variance_samples
        Q = len(self._functional_kernel.kernels)
        par = min(max(self.max_procs, 1), max(Ns, Q))
        W = self.interpolant
        ls = [(W, coreg_mat, toep.top, np.random.randn(W.shape[1], Ns), i)
              for i, (coreg_mat, toep) in
              enumerate(zip(self._functional_kernel.coreg_mats(),
                            self.kernel.materialized_kernels))]
        _LOG.info('Using %d processors to '
                  'precompute %d kernel factors',
                  par, Q)
        with closing(Pool(processes=par)) as pool:
            samples = pool.starmap(InterpolatedLLGP._chol_sample, ls)
            samples.append(
                Diag(np.sqrt(self.kernel.K.noise.v)).matmat(
                    np.random.randn(len(self.y), Ns)))
            samples = np.array(samples).sum(axis=0).T

            # Re-use same pool
            _LOG.info('Using %d processors to precompute %d '
                      'variance samples', par, Ns)
            ls = [(self.kernel.K, sample) for sample in samples]
            samples = np.array(pool.starmap(Iterative.solve, ls)).T

        nu = np.square(self.kernel.K.grid_K.matmat(
            self.interpolantT.dot(samples))).sum(axis=1) / Ns

        self._cache['sampled_nu'] = nu
        return nu

    def _var_predict_sample(self, prediction_interpolant, _):
        nu = self._sampled_nu()
        return prediction_interpolant.dot(nu)

    @staticmethod
    def _var_solve(i, K, K_XU, K_UX):
        x = np.zeros(K_XU.shape[1])
        x[i] = 1
        x = K_XU.matvec(x)
        x = Iterative.solve(K, x)
        x = K_UX.matvec(x)
        return x[i]

    def _precomputed_nu(self):
        if 'precomputed_nu' in self._cache:
            return self._cache['precomputed_nu']

        W = self.interpolant
        WT = self.interpolantT
        K_UU = self.kernel.K.grid_K
        # SKI should have a half_matrix() method returning these two,
        # in which case we'd just ask self.kernel.K.ski for them.
        K_XU = Composition([Matrix.wrap(W.shape, W.dot), K_UU])
        K_UX = Composition([K_UU, Matrix.wrap(WT.shape, WT.dot)])

        m = len(self.inducing_grid)
        D = len(self._functional_kernel.noise)
        Dm = D * m
        assert Dm == K_XU.shape[1] and Dm == K_UX.shape[0]

        par = min(max(self.max_procs, 1), Dm)
        chunks = max(K_XU.shape[1] // par // 4, 1)
        ls = [(i, self.kernel.K, K_XU, K_UX) for i in range(Dm)]
        _LOG.info('Using %d processors to precompute %d '
                  'variance terms exactly', par, Dm)
        with closing(Pool(processes=par)) as pool:
            nu = pool.starmap(InterpolatedLLGP._var_solve, ls, chunks)

        self._cache['precomputed_nu'] = nu
        return nu

    def _var_predict_precompute(self, prediction_interpolant, _):
        nu = self._precomputed_nu()
        return prediction_interpolant.dot(nu)

    def _var_predict_on_the_fly(self, _, Xs):
        K_test_X = ExactLMCLikelihood.kernel_from_indices(
            Xs, self.Xs, self._functional_kernel)
        n_test = sum(map(len, Xs))
        par = min(max(self.max_procs, 1), n_test)
        _LOG.info('Using %d processors for %d on-the-fly variance'
                  ' predictions', par, n_test)
        ls = [(self.kernel.K, k_star) for k_star in K_test_X]
        with closing(Pool(processes=par)) as pool:
            inverted = np.array(pool.starmap(Iterative.solve, ls)).T

        full_mat = K_test_X.dot(inverted)
        return np.diag(full_mat)
