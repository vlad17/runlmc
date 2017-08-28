# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

from contextlib import closing
import logging
from multiprocessing import Pool, cpu_count
import os

import numpy as np
import scipy.linalg as la
import scipy.spatial.distance as dist
import scipy.stats
from paramz.transformations import Logexp

from .multigp import MultiGP
from ..approx.interpolation import multi_interpolant, autogrid
from ..approx.iterative import Iterative
from ..linalg.composition import Composition
from ..linalg.matrix import Matrix
from ..linalg.numpy_matrix import NumpyMatrix
from ..linalg.kronecker import Kronecker
from ..linalg.diag import Diag
from ..lmc.stochastic_deriv import StochasticDeriv
from ..lmc.parameter_values import ParameterValues
from ..lmc.grid_kernel import gen_grid_kernel
from ..lmc.metrics import Metrics
from ..lmc.kernel import ExactLMCKernel, ApproxLMCKernel
from ..parameterization.param import Param
from ..util.docs import inherit_doc

_LOG = logging.getLogger(__name__)


@inherit_doc  # pylint: disable=too-many-instance-attributes
class LMC(MultiGP):
    """
    The main class of this package, `LMC` implements linearithmic
    Gaussian Process learning in the multi-output case. See
    the paper on `arxiv <https://arxiv.org/abs/1705.10813>`_.

    .. Note: Currently, only one-dimensional input is supported.

    Upon construction, this class assumes ownership of its parameters and
    does not account for changes in their values.

    The exact kernel that this approximates is the following:

    .. math::

        K_{\\text{exact}}=\sum_{q=1}^Q\\left(A_qA_q^\\top+
             \\boldsymbol\\kappa_q I\\right)
             \circ [k_q(X_i, X_j)]_{ij\in[D]^2} +
             \\boldsymbol\epsilon

    :math:`[\cdot]_{ij}` represents a block matrix, with rows and columns
    possibly of different widths. :math:`\circ` is the Hadamard product.
    :math:`\\boldsymbol\\epsilon` is a diagonal Gaussian noise
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
    :param prediction: one of `'matrix-free'`, `'on-the-fly'`,
                              `'precompute'`, `'exact'`, `'sample'`.
    :param variance_samples: only used if `prediction` is set to `'sample'`.
                             number of variance samples used for predictive
                             variance.
    :param max_procs: maximum number of processes to use for parallelism,
                      defaults to cpu count.
    :param slfm_kernels: add kernel terms with kernel :math:`k_q` given by this
                       list
                       using the SLFM model, which means the corresponding rank
                       is 1 and terms :math:`\\boldsymbol\\kappa_q=\\textbf{0}`
    :param indep_gp: add in a term for independent GPs for each output,
                     this must be a list of :math:`D` kernels if specified
                     at all, corresponding to terms with
                     :math:`A_q=0,\\boldsymbol\\kappa_q=\\boldsymbol\\kron_d`
                     for the :math:`d`-th element of this list
                     :math:`k_d`.
    :raises: :class:`ValueError` if `Xs` and `Ys` lengths do not match.
    :raises: :class:`ValueError` if normalization if any `Ys` have no variance
                                 or values in `Xs` have multiple identical
                                 values.
    :raises: :class:`ValueError` if no kernels
    """

    def __init__(self, Xs, Ys, normalize=True, kernels=[],  # pylint: disable=too-many-arguments,dangerous-default-value,too-many-locals
                 ranks=None, lo=None, hi=None, m=None, name='lmc',
                 metrics=False, prediction='matrix-free',
                 variance_samples=20, max_procs=None,
                 slfm_kernels=[], indep_gp=[]):
        super().__init__(Xs, Ys, normalize=normalize, name=name)
        self.update_model(False)
        # TODO(cleanup) - this entire constructor needs reorg, refactor
        # into smaller methods, etc. Large number of arguments is OK, though.
        # basically address all pylint complaints (disabled in the constructor)

        if not kernels and not slfm_kernels and not indep_gp:
            raise ValueError('Number of kernels should be >0')

        if metrics and (slfm_kernels or indep_gp):
            raise ValueError('Metrics incompatible with slfm/indep gp')

        if len(indep_gp) not in [0, len(Xs)]:
            raise ValueError('Independent GP kernels should be one-per-output'
                             ' or not specified at all')

        self.variance_samples = variance_samples
        self.prediction = prediction
        if prediction not in self._prediction_methods():
            raise ValueError('Variance prediction method {} unrecognized'
                             .format(prediction))

        self.kernels = kernels + slfm_kernels + indep_gp
        self.nkernels = {
            'lmc': len(kernels),
            'slfm': len(slfm_kernels),
            'indep': len(indep_gp)}
        for k in self.kernels:
            self.link_parameter(k)

        n = sum(map(len, self.Xs))
        _LOG.info('LMC %s generating inducing grid n = %d',
                  self.name, n)
        # Grid corresponds to U
        self.inducing_grid, m = autogrid(Xs, lo, hi, m)

        # Toeplitz(self.dists) is the pairwise distance matrix of U
        self.dists = self.inducing_grid - self.inducing_grid[0]

        # Corresponds to W; block diagonal matrix.
        self.interpolant = multi_interpolant(self.Xs, self.inducing_grid)
        self.interpolantT = self.interpolant.transpose().tocsr()

        _LOG.info('LMC %s grid (n = %d, m = %d) complete, ',
                  self.name, n, m)

        if ranks is None:
            ranks = [1 for _ in kernels]

        distrib = scipy.stats.truncnorm(-1, 1)

        def randinit(sx, sy):
            return distrib.rvs(size=(sx, sy))

        self.coreg_vecs = []
        initial_vecs = []
        initial_vecs += [randinit(rank, self.output_dim) for rank in ranks]
        initial_vecs += [randinit(1, self.output_dim)
                         for _ in slfm_kernels]
        initial_vecs += [np.zeros((1, self.output_dim)) for _ in indep_gp]
        for i, coreg_vec in enumerate(initial_vecs):
            self.coreg_vecs.append(Param('a{}'.format(i), coreg_vec))
            if i < self.nkernels['lmc'] + self.nkernels['slfm']:
                self.link_parameter(self.coreg_vecs[-1])

        self.coreg_diags = []
        for _ in range(self.nkernels['lmc']):
            i = len(self.coreg_diags)
            coreg_diags = np.ones(self.output_dim)
            self.coreg_diags.append(
                Param('kappa{}'.format(i), coreg_diags, Logexp()))
            self.link_parameter(self.coreg_diags[-1])
        for _ in range(self.nkernels['slfm']):
            i = len(self.coreg_diags)
            coreg_diags = np.zeros(self.output_dim)
            self.coreg_diags.append(Param('kappa{}'.format(i), coreg_diags))
            self.coreg_diags[-1].constrain_fixed()
        for d in range(self.nkernels['indep']):
            i = len(self.coreg_diags)
            coreg_diags = np.zeros(self.output_dim)
            coreg_diags[d] = 1
            self.coreg_diags.append(Param('kappa{}'.format(i), coreg_diags))
            self.coreg_diags[-1].constrain_fixed()

        # Corresponds to epsilon
        self.noise = Param('noise', 0.1 * np.ones(self.output_dim), Logexp())
        self.link_parameter(self.noise)

        self.y = np.hstack(self.Ys)

        self.kernel = None
        self._cache = {}
        self.metrics = Metrics() if metrics else None
        self.max_procs = cpu_count() if max_procs is None else max_procs
        self._pool = None

        self._check_omp(max_procs)

        self.update_model(True)
        _LOG.info('LMC %s fully initialized', self.name)

    EVAL_NORM = np.inf

    def _check_omp(self, procs_requested):
        omp = os.environ.get('OMP_NUM_THREADS', None)
        procs_info = 'LMC(max_procs={})'.format(procs_requested)
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

        params = ParameterValues.generate(self)
        grid_kernel = gen_grid_kernel(
            params, self.dists, self.interpolant, self.interpolantT)
        self.kernel = ApproxLMCKernel(
            params, grid_kernel, self.dists, self.metrics, self._pool)

        if _LOG.isEnabledFor(logging.DEBUG):
            fmt = '{:7.6e}'.format

            def np_print(x):
                return np.array2string(np.copy(x), formatter={'float': fmt})
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
        for x, dx in zip(self.coreg_diags,
                         self.kernel.coreg_diags_gradients()):
            x.gradient = dx
        for k, dk in zip(self.kernels, self.kernel.kernel_gradients()):
            k.update_gradient(dk)
        self.noise.gradient = self.kernel.noise_gradient()

        if self.metrics is not None:
            grad_norm = la.norm(self.gradient, self.EVAL_NORM)
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
                np.concatenate(exact.coreg_diags_gradients()),
                np.concatenate(exact.kernel_gradients()),
                exact.noise_gradient()))

            self.metrics.grad_norms.append(grad_norm)
            self.metrics.grad_error.append(
                la.norm(ordered_grad - exact_grad, self.EVAL_NORM)
                / la.norm(exact_grad, self.EVAL_NORM))
            self.metrics.log_likely.append(self.log_likelihood())

    def _dense(self):
        if 'exact_kernel' not in self._cache:
            pdists = dist.pdist(np.hstack(self.Xs).reshape(-1, 1))
            pdists = dist.squareform(pdists)
            self._cache['exact_kernel'] = ExactLMCKernel(
                ParameterValues.generate(self), pdists)
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
                                     for per_output in self.coreg_vecs)
            coregs += np.column_stack(self.coreg_diags)
            kernels = [k.from_dist(0) for k in self.kernels]
            native_output_var = coregs.dot(kernels).reshape(-1)
            native_var = native_output_var + self.noise

            self._cache['native_var'] = native_var
        return self._cache['native_var']

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
        tot_grid_size = len(self.inducing_grid) * len(self.noise)
        if tot_pred_size > tot_grid_size:
            return self._var_predict_precompute(W, Xs)
        return self._var_predict_on_the_fly(W, Xs)

    def _var_predict_exact(self, _, Xs):
        # TODO(cleanup) refactor ExactLMCKernel so that we can reuse code
        # without the ugly construction and params.lens_x stuff used here.
        exact = self._dense()

        test_Xs = np.hstack(Xs).reshape(-1, 1)
        train_Xs = np.hstack(self.Xs).reshape(-1, 1)
        params = ParameterValues.generate(self)
        params.lens_x = [len(X) for X in Xs]
        K_test_X = ExactLMCKernel(params,
                                  dist.cdist(test_Xs, train_Xs),
                                  invert=False, noise=False).K
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
        Q = len(self.kernels)
        par = min(max(self.max_procs, 1), max(Ns, Q))
        W = self.interpolant
        ls = [(W, coreg_mat, toep.top, np.random.randn(W.shape[1], Ns), i)
              for i, (coreg_mat, toep) in
              enumerate(zip(self.kernel.params.coreg_mats,
                            self.kernel.materialized_kernels))]
        _LOG.info('Using %d processors to '
                  'precompute %d kernel factors',
                  par, Q)
        with closing(Pool(processes=par)) as pool:
            samples = pool.starmap(LMC._chol_sample, ls)
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
        D = len(self.noise)
        Dm = D * m
        assert Dm == K_XU.shape[1] and Dm == K_UX.shape[0]

        par = min(max(self.max_procs, 1), Dm)
        chunks = max(K_XU.shape[1] // par // 4, 1)
        ls = [(i, self.kernel.K, K_XU, K_UX) for i in range(Dm)]
        _LOG.info('Using %d processors to precompute %d '
                  'variance terms exactly', par, Dm)
        with closing(Pool(processes=par)) as pool:
            nu = pool.starmap(LMC._var_solve, ls, chunks)

        self._cache['precomputed_nu'] = nu
        return nu

    def _var_predict_precompute(self, prediction_interpolant, _):
        nu = self._precomputed_nu()
        return prediction_interpolant.dot(nu)

    def _var_predict_on_the_fly(self, _, Xs):
        # TODO(cleanup) refactor ExactLMCKernel so that we can reuse code
        # without the ugly construction and params.lens_x stuff used here.

        test_Xs = np.hstack(Xs).reshape(-1, 1)
        train_Xs = np.hstack(self.Xs).reshape(-1, 1)
        params = ParameterValues.generate(self)
        params.lens_x = [len(X) for X in Xs]
        K_test_X = ExactLMCKernel(params,
                                  dist.cdist(test_Xs, train_Xs),
                                  invert=False, noise=False).K
        n_test = test_Xs.shape[0]
        par = min(max(self.max_procs, 1), n_test)
        _LOG.info('Using %d processors for %d on-the-fly variance'
                  ' predictions', par, n_test)
        ls = [(self.kernel.K, k_star) for k_star in K_test_X]
        with closing(Pool(processes=par)) as pool:
            inverted = np.array(pool.starmap(Iterative.solve, ls)).T

        return np.diag(K_test_X.dot(inverted))
