# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import itertools
from functools import partial

import numpy as np
from parameterized import parameterized
import scipy.linalg as la
import scipy.spatial.distance as dist

from .interpolated_llgp import InterpolatedLLGP
from .optimization import AdaDelta
from ..kern.rbf import RBF
from ..lmc.likelihood import ExactLMCLikelihood
from ..lmc.functional_kernel import FunctionalKernel
from ..util.numpy_convenience import tesselate
from ..util.testing_utils import RandomTest, check_np_lists, vectorize_inputs


class ExactAnalogue:
    def __init__(self, kernels, sizes, coregs, xss=None, yss=None, indim=1):
        assert len(coregs) == len(kernels)
        assert set(map(lambda x: x.shape[1], coregs)) == {len(sizes)}
        for coreg in coregs:
            assert coreg.shape[0] <= len(sizes), (coreg.shape, len(sizes))

        diags = [np.ones(len(sizes)) for _ in coregs]
        noise = np.ones(len(sizes))
        if yss is None:
            yss = [np.random.rand(sz) for sz in sizes]
        if xss is None:
            if indim is None:  # explicitly test 1d shorthand
                xss = [np.random.rand(sz) for sz in sizes]
            else:
                xss = [np.random.rand(sz, indim) for sz in sizes]

        self.xss, self.yss = xss, yss
        self.lens = sizes
        ranks = [len(x) for x in coregs]
        fk = FunctionalKernel(D=len(xss), lmc_kernels=kernels,
                              lmc_ranks=ranks)
        for lmc_coreg, coreg in zip(fk._coreg_vecs, coregs):
            lmc_coreg[:] = coreg
        for lmc_coreg, coreg in zip(fk._coreg_diags, diags):
            lmc_coreg[:] = coreg
        fk._noise[:] = noise
        self.kernels = kernels
        self.functional_kernel = fk
        self.exact = None
        self.indim = indim or 1

    def gen_lmc(self, m):
        lmc = InterpolatedLLGP(self.xss, self.yss, normalize=False, m=m,
                               functional_kernel=self.functional_kernel)
        return lmc

    def gen_exact(self):
        if self.exact is None:
            xss = [xs if xs.ndim == 2 else xs.reshape(-1, 1)
                   for xs in self.xss]
            self.functional_kernel.set_input_dim(self.indim)
            self.exact = ExactLMCLikelihood(
                self.functional_kernel, xss, self.yss)

        return self.exact

    @staticmethod
    def pairwise_dists(kern, xs1, xs2):
        dists = dist.cdist(
            xs1.reshape(-1, 1), xs2.reshape(-1, 1))
        return kern.from_dist(dists).reshape(len(xs1), len(xs2))

    @staticmethod
    def gen_obs(xss, noise_sd, true_func):
        true_func = [vectorize_inputs(f) for f in true_func]
        assert all(x <= 0.1 for x in noise_sd)
        assert len(noise_sd) == len(true_func)
        xss = [xs.reshape(-1, 1) if xs.ndim == 1 else xs for xs in xss]
        noises = [np.random.randn(len(xs)) * sd
                  for xs, sd in zip(xss, noise_sd)]
        yss = [f(xs).ravel() + noise
               for f, xs, noise in zip(true_func, xss, noises)]
        return yss


class LMCTestUtils:

    @staticmethod
    def _case_1d(input_dim):
        kernels = [RBF(inv_lengthscale=3)]
        szs = [10]
        coregs = [np.array([[1]])]
        return ExactAnalogue(kernels, szs, coregs, indim=input_dim)

    @staticmethod
    def _case_2d(input_dim):
        kernels = [RBF(inv_lengthscale=3),
                   RBF(inv_lengthscale=2)]
        szs = [10, 15]
        coregs = [np.array(x).reshape(1, -1) for x in [[1, 2], [3, 4]]]
        return ExactAnalogue(kernels, szs, coregs, indim=input_dim)

    @staticmethod
    def _case_2d_splitkern(split1, split2, input_dim):
        assert input_dim >= 2
        kernels = [RBF(inv_lengthscale=3, active_dims=split1),
                   RBF(inv_lengthscale=2, active_dims=split2)]
        szs = [10, 15]
        coregs = [np.array(x).reshape(1, -1) for x in [[1, 2], [3, 4]]]
        return ExactAnalogue(kernels, szs, coregs, indim=input_dim)

    @staticmethod
    def _case_multirank(input_dim):
        kernels = [RBF(inv_lengthscale=3),
                   RBF(inv_lengthscale=2)]
        szs = [10, 15]
        coregs = [np.array([[1, 2], [3, 4]]), np.array([[1, 1]])]
        return ExactAnalogue(kernels, szs, coregs, indim=input_dim)

    @staticmethod
    def _case_large(input_dim):
        kernels = [RBF(inv_lengthscale=3),
                   RBF(inv_lengthscale=2),
                   RBF(inv_lengthscale=1)]
        szs = [8, 8, 8, 8, 8]
        coregs = [np.array(x).reshape(1, -1) for x in
                  [[1, 1, 1, 1, 2], [2, 1, 2, 1, 2], [-1, 1, -1, -1, -1]]]
        return ExactAnalogue(kernels, szs, coregs, indim=input_dim)

    @staticmethod
    def avg_entry_diff(x1, x2):
        return np.fabs(x1 - x2).mean()

    @classmethod
    def _output_cases(cls):
        return {
            'output_1d': cls._case_1d,
            'output_2d': cls._case_2d,
            'output_multirank': cls._case_multirank,
            'output_large': cls._case_large}

    @classmethod
    def _input_cases(cls):
        return {
            'input_1dimplicit': None,
            'input_1d': 1,
            'input_2d': 2,
            'input_3d': 3}

    @classmethod
    def _input_splits(cls, in_dim):
        if in_dim is None or in_dim < 2:
            return []
        elif in_dim == 2:
            return [((0, 1), (0,)), ((0,), (1,))]
        assert in_dim == 3, in_dim
        return [((0, 1), (2,)), ((0, 1), (1, 2)), ((0,), (1,)), ((0, 1), (0,))]

    @classmethod
    def input_output_cases_grid(cls):
        outs = cls._output_cases()
        ins = cls._input_cases()
        for (out_name, out), (in_name, in_dim) in itertools.product(
                outs.items(), ins.items()):
            # Non-split kernels: no more than two input dims
            if in_dim is not None and in_dim > 2:
                continue
            yield out_name + '_' + in_name, out, in_dim
        for in_name, in_dim in ins.items():
            for splits in cls._input_splits(in_dim):
                name = 'output_2d_input_2d_splitkern_'
                name += '_'.join(map(str, splits[0]))
                name += '___'
                name += '_'.join(map(str, splits[1]))
                case = partial(cls._case_2d_splitkern, splits[0], splits[1])
                yield name, case, in_dim


class LMCTest(RandomTest):

    def _check_kernel_reconstruction(self, exact, indim):
        def reconstruct(x):
            return x.kernel.K.as_numpy()
        m = sum(exact.lens) * np.ones(indim) * 2
        actual = reconstruct(exact.gen_lmc(m))
        exact_mat = exact.gen_exact().K
        tol = 1e-3
        np.testing.assert_allclose(
            exact_mat, actual, rtol=tol, atol=tol)
        avg_diff_sz = LMCTestUtils.avg_entry_diff(exact_mat, actual)

        actual = reconstruct(exact.gen_lmc(m * 2))
        np.testing.assert_allclose(
            exact_mat, actual, rtol=tol, atol=tol)
        avg_diff_2sz = LMCTestUtils.avg_entry_diff(exact_mat, actual)

        self.assertGreater(avg_diff_sz, avg_diff_2sz)

        xss = [xs if xs.ndim == 2 else xs.reshape(-1, 1)
               for xs in exact.xss]

        # Also check ExactLMCLikelihood reconstruction
        K_X_X = ExactLMCLikelihood.kernel_from_indices(
            xss, xss, exact.functional_kernel) + np.diag(
                np.repeat(exact.functional_kernel.noise, exact.lens))
        np.testing.assert_allclose(exact_mat, K_X_X, rtol=tol, atol=tol)

    def _check_kernels_equal(self, tol, a, b, check_gradients=True):
        np.testing.assert_allclose(
            a.alpha(), b.alpha(), tol, tol)
        if check_gradients:
            check_np_lists(
                a.coreg_vec_gradients(), b.coreg_vec_gradients(),
                atol=tol, rtol=tol)
            check_np_lists(
                a.coreg_diags_gradients(), b.coreg_diags_gradients(),
                atol=tol, rtol=tol)
            check_np_lists(
                a.kernel_gradients(), b.kernel_gradients(),
                atol=tol, rtol=tol)
            np.testing.assert_allclose(
                a.noise_gradient(), b.noise_gradient(), tol, tol)

    def _check_kernel_params(self, ea, input_dim):
        m = sum(ea.lens) / len(ea.lens) * np.ones(input_dim)
        actual = ea.gen_lmc(m)
        exact = ea.gen_exact()

        tol = 1e-3
        self._check_kernels_equal(tol, exact, actual._dense())
        self._check_kernels_equal(
            tol, exact, actual.kernel, check_gradients=False)

    def _check_normal_quadratic(self, exact, input_dim):
        exact_mat = exact.gen_exact().K
        y = np.hstack(exact.yss)
        Kinv_y = la.solve(exact_mat, y)
        expected = y.dot(Kinv_y)

        lmc = exact.gen_lmc(sum(exact.lens) * np.ones(input_dim) * 2)
        lmc.TOL = 1e-15  # tighten tolerance for tests
        tol = 1e-3

        actual = lmc.normal_quadratic()
        np.testing.assert_allclose(expected, actual, rtol=tol, atol=tol)

    def _check_fit(self, ea, input_dim):
        lmc = ea.gen_lmc(sum(ea.lens) * np.ones(input_dim) * 2)

        ll_before = lmc.log_likelihood()
        lmc.optimize(optimizer=AdaDelta(max_it=5))
        ll_after = lmc.log_likelihood()

        self.assertGreater(ll_after, ll_before)

    def _check_prediction(self, ea, input_dim):  # pylint: disable=too-many-locals
        lmc = ea.gen_lmc(sum(ea.lens) * np.ones(input_dim) * 2)

        # Apply formula for conditional Gaussian

        K = ea.gen_exact().K
        input_dim = input_dim or 1
        test_xss = [np.random.rand(5, input_dim) for _ in ea.xss]
        test_lens = [len(xs) for xs in test_xss]
        xss = [xs if xs.ndim == 2 else xs.reshape(-1, 1)
               for xs in ea.xss]
        K_test_X = ExactLMCLikelihood.kernel_from_indices(
            test_xss, xss, ea.functional_kernel)
        expected_mean = K_test_X.dot(la.solve(K, np.hstack(ea.yss)))
        expected_mean = tesselate(expected_mean, test_lens)

        K_test_test = ExactLMCLikelihood.kernel_from_indices(
            test_xss, test_xss, ea.functional_kernel) + np.diag(
                np.repeat(ea.functional_kernel.noise, test_lens))
        expected_var = K_test_test - K_test_X.dot(la.solve(K, K_test_X.T))
        expected_var = tesselate(np.diag(expected_var), test_lens)

        lmc.prediction = 'exact'
        actual_exact_mean, actual_exact_var = lmc.predict(test_xss)
        # The O(1) mean can actually be fairly inaccurate
        np.testing.assert_allclose(expected_mean, actual_exact_mean,
                                   rtol=1e-1, atol=1e-1)
        np.testing.assert_allclose(
            expected_var, actual_exact_var, rtol=1e-4, atol=1e-4)

        lmc.prediction = 'on-the-fly'
        lmc.TOL = 1e-15  # tighten tolerance for tests
        actual_mean, actual_var = lmc.predict(test_xss)
        np.testing.assert_allclose(actual_exact_mean, actual_mean,
                                   rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(
            expected_var, actual_var, rtol=1e-4, atol=1e-4)

    def test_no_kernel(self):
        def mapnp(x):
            return list(map(np.array, x))
        basic_Xs = mapnp([[0, 1, 2], [0.5, 1.5, 2.5]])
        basic_Ys = mapnp([[5, 6, 7], [7, 6, 5]])
        self.assertRaises(ValueError, InterpolatedLLGP, basic_Xs, basic_Ys)

    @parameterized.expand(LMCTestUtils.input_output_cases_grid())
    def test_kernel_reconstruction(self, _, output_case, input_dim):
        ea = output_case(input_dim)
        self._check_kernel_reconstruction(ea, input_dim)

    @parameterized.expand(LMCTestUtils.input_output_cases_grid())
    def test_kernel_params(self, _, output_case, input_dim):
        ea = output_case(input_dim)
        self._check_kernel_params(ea, input_dim)

    @parameterized.expand(LMCTestUtils.input_output_cases_grid())
    def test_normal_quadratic(self, _, output_case, input_dim):
        ea = output_case(input_dim)
        self._check_normal_quadratic(ea, input_dim)

    @parameterized.expand(LMCTestUtils.input_output_cases_grid())
    def test_prediction(self, _, output_case, input_dim):
        ea = output_case(input_dim)
        self._check_prediction(ea, input_dim)

    @parameterized.expand([
        ('output_1d_input1d', LMCTestUtils._case_1d, 1, [0.05], [np.sin]),
        ('output_1d_input1dimplicit',
         LMCTestUtils._case_1d, None, [0.05], [np.sin]),
        ('output_1d_input2d', LMCTestUtils._case_1d, 2, [0.05], [
            lambda x, y: np.sin(x - y)]),
        ('output_2d', LMCTestUtils._case_2d,
         1, [0.05, 0.08], [np.sin, np.cos]),
        ('output_2d_noisediff', LMCTestUtils._case_2d,
         1, [1e-8, 0.08], [np.sin, np.cos]),
        ('output_2d_input2d', LMCTestUtils._case_2d, 2, [0.05, 0.08], [
            (lambda x, y: np.sin(x - y)),
            (lambda x, y: np.cos(np.exp(x) / (np.abs(y) + 1)))]),
        ('output_multirank', LMCTestUtils._case_multirank,
         1, [0.05, 0.08], [np.sin, np.cos]),
        ('output_large', LMCTestUtils._case_large, 1, [0.05] * 5, [
            np.sin, np.cos, np.exp, np.sin, np.cos])
    ])
    def test_fit(self, _, output_case, input_dim, noise_sd, true_func):
        ea = output_case(input_dim)
        yss = ExactAnalogue.gen_obs(ea.xss, noise_sd, true_func)
        ea = ExactAnalogue(ea.kernels, ea.lens,
                           ea.functional_kernel.coreg_vecs, ea.xss, yss)
        self._check_fit(ea, input_dim)

    def test_2d_1k_fit_large_offset(self):
        kernels = [RBF(inv_lengthscale=3)]
        szs = [30, 40]
        coregs = [np.array([[1, 1]])]
        ea = ExactAnalogue(kernels, szs, coregs)
        noise_sd = [0.02, 0.08]
        true_func = [np.sin, lambda x: np.cos(x) + 100]
        yss = ExactAnalogue.gen_obs(ea.xss, noise_sd, true_func)
        ea = ExactAnalogue(ea.kernels, ea.lens,
                           ea.functional_kernel.coreg_vecs, ea.xss, yss)
        self._check_fit(ea, None)
