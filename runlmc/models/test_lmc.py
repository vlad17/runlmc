# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import unittest

import numpy as np
import scipy.spatial.distance as dist

from .lmc import LMC
from .optimization import AdaDelta
from ..kern.rbf import RBF
from ..lmc.parameter_values import ParameterValues
from ..lmc.kernel import ExactLMCKernel
from ..util.testing_utils import RandomTest

class ExactAnalogue:
    def __init__(self, kerns, sizes, coregs, xss=None, yss=None):
        assert len(coregs) == len(kerns)
        assert set(map(len, coregs)) == {len(sizes)}

        diags = [np.ones(len(sizes)) for _ in coregs]
        noise = np.ones(len(sizes))
        if yss is None:
            yss = [np.random.rand(sz) for sz in sizes]
        if xss is None:
            xss = [np.random.rand(sz) for sz in sizes]

        self.xss, self.yss = xss, yss
        pdists = dist.pdist(np.hstack(xss).reshape(-1, 1))
        self.pdists = dist.squareform(pdists)

        self.params = ParameterValues(
            coregs, diags, kerns, sizes, np.hstack(yss), noise)
        self.exact = None

    def gen_lmc(self, m):
        lmc = LMC(self.xss, self.yss,
                  normalize=False, kernels=self.params.kernels, m=m)
        for lmc_coreg, coreg in zip(lmc.coreg_vecs, self.params.coreg_vecs):
            lmc_coreg[:] = coreg
        for lmc_coreg, coreg in zip(lmc.coreg_diags, self.params.coreg_diag):
            lmc_coreg[:] = coreg
        lmc.noise[:] = self.params.noise
        return lmc

    def gen_exact(self):
        if self.exact is None:
            self.exact = ExactLMCKernel(self.params, self.pdists)

        return self.exact

    @staticmethod
    def pairwise_dists(kern, xs1, xs2):
        dists = dist.cdist(
            xs1.reshape(-1, 1), xs2.reshape(-1, 1))
        return kern.from_dist(dists).reshape(len(xs1), len(xs2))

    @staticmethod
    def gen_obs(xss, noise_sd, true_func):
        assert all(x <= 0.1 for x in noise_sd)
        assert len(noise_sd) == len(true_func)
        noises = [np.random.randn(len(xs)) * sd
                  for xs, sd in zip(xss, noise_sd)]
        yss = [f(xs) + noise
               for f, xs, noise in zip(true_func, xss, noises)]
        return yss

class LMCTest(RandomTest):

    def setUp(self):
        super().setUp()

    @staticmethod
    def case_1d():
        kerns = [RBF(inv_lengthscale=3)]
        szs = [30]
        coregs = [[1]]
        return ExactAnalogue(kerns, szs, coregs)

    @staticmethod
    def case_2d():
        kerns = [RBF(inv_lengthscale=3),
                 RBF(inv_lengthscale=2)]
        szs = [30, 40]
        coregs = [[1, 2], [3, 4]]
        return ExactAnalogue(kerns, szs, coregs)

    @staticmethod
    def case_large():
        kerns = [RBF(inv_lengthscale=3),
                 RBF(inv_lengthscale=2),
                 RBF(inv_lengthscale=1)]
        szs = [10, 12, 14, 12, 10]
        coregs = [[1, 1, 1, 1, 2], [2, 1, 2, 1, 2], [-1, 1, -1, -1, -1]]
        return ExactAnalogue(kerns, szs, coregs)

    @staticmethod
    def avg_entry_diff(x1, x2):
        return np.fabs(x1 - x2).mean()

    def check_kernel_reconstruction(self, exact):
        reconstruct = lambda x: x.kernel.K.as_numpy()
        actual = reconstruct(exact.gen_lmc(sum(exact.params.lens)))
        exact_mat = exact.gen_exact().K
        tol = 1e-4
        np.testing.assert_allclose(
            exact_mat, actual, rtol=tol, atol=tol)
        avg_diff_sz = self.avg_entry_diff(exact_mat, actual)

        actual = reconstruct(exact.gen_lmc(sum(exact.params.lens) * 2))
        np.testing.assert_allclose(
            exact_mat, actual, rtol=tol, atol=tol)
        avg_diff_2sz = self.avg_entry_diff(exact_mat, actual)

        self.assertGreater(avg_diff_sz, avg_diff_2sz)

    def check_normal_quadratic(self, exact):
        exact_mat = exact.gen_exact().K
        y = np.hstack(exact.yss)
        Kinv_y = np.linalg.solve(exact_mat, y)
        expected = y.dot(Kinv_y)

        lmc = exact.gen_lmc(sum(exact.params.lens))
        lmc.TOL = 1e-15 # tighten tolerance for tests
        tol = 1e-4

        actual = lmc.normal_quadratic()
        np.testing.assert_allclose(expected, actual, rtol=tol, atol=tol)

    def check_fit(self, ea):
        lmc = ea.gen_lmc(sum(ea.params.lens))

        ll_before = lmc.log_likelihood()
        lmc.optimize(optimizer=AdaDelta(max_it=5))
        ll_after = lmc.log_likelihood()

        self.assertGreater(ll_after, ll_before)

    def test_no_kernel(self):
        mapnp = lambda x: list(map(np.array, x))
        basic_Xs = mapnp([[0, 1, 2], [0.5, 1.5, 2.5]])
        basic_Ys = mapnp([[5, 6, 7], [7, 6, 5]])
        self.assertRaises(ValueError, LMC,
                          basic_Xs, basic_Ys, kernels=[])

    def test_kernel_reconstruction_1d(self):
        ea = self.case_1d()
        self.check_kernel_reconstruction(ea)

    def test_kernel_reconstruction_2d(self):
        ea = self.case_2d()
        self.check_kernel_reconstruction(ea)

    def test_kernel_reconstruction_large(self):
        ea = self.case_large()
        self.check_kernel_reconstruction(ea)

    def test_normal_quadratic_1d(self):
        ea = self.case_1d()
        self.check_normal_quadratic(ea)

    def test_normal_quadratic_2d(self):
        ea = self.case_2d()
        self.check_normal_quadratic(ea)

    def test_normal_quadratic_large(self):
        ea = self.case_large()
        self.check_normal_quadratic(ea)

    def test_1d_fit(self):
        ea = self.case_1d()
        noise_sd = [0.05]
        true_func = [np.sin]
        yss = ExactAnalogue.gen_obs(ea.xss, noise_sd, true_func)
        ea = ExactAnalogue(ea.params.kernels, ea.params.lens,
                           ea.params.coreg_vecs, ea.xss, yss)
        self.check_fit(ea)

    def test_2d_fit(self):
        ea = self.case_2d()
        noise_sd = [0.05, 0.08]
        true_func = [np.sin, np.cos]
        yss = ExactAnalogue.gen_obs(ea.xss, noise_sd, true_func)
        ea = ExactAnalogue(ea.params.kernels, ea.params.lens,
                           ea.params.coreg_vecs, ea.xss, yss)
        self.check_fit(ea)

    def test_2d_fit_noisediff(self):
        ea = self.case_2d()
        noise_sd = [1e-8, 0.09]
        true_func = [np.sin, np.cos]
        yss = ExactAnalogue.gen_obs(ea.xss, noise_sd, true_func)
        ea = ExactAnalogue(ea.params.kernels, ea.params.lens,
                           ea.params.coreg_vecs, ea.xss, yss)
        self.check_fit(ea)

    def test_2d_1k_fit_large_offset(self):
        kerns = [RBF(inv_lengthscale=3)]
        szs = [30, 40]
        coregs = [[1, 1]]
        ea = ExactAnalogue(kerns, szs, coregs)
        noise_sd = [0.02, 0.08]
        true_func = [np.sin, lambda x: np.cos(x) + 100]
        yss = ExactAnalogue.gen_obs(ea.xss, noise_sd, true_func)
        ea = ExactAnalogue(ea.params.kernels, ea.params.lens,
                           ea.params.coreg_vecs, ea.xss, yss)
        self.check_fit(ea)
