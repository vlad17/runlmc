# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import unittest

import numpy as np
# TODO(cleanup): optimizer with import paramz.optimization
import scipy.linalg
import scipy.optimize
import scipy.spatial.distance

from .lmc import LMC
from ..kern.rbf import RBF
from ..util.testing_utils import RandomTest

class ExactAnalogue:
    def __init__(self, kerns, sizes, coregs, diags=None):
        assert len(coregs) == len(kerns)
        assert set(map(len, coregs)) == {len(sizes)}
        self.kerns = kerns
        self.sizes = sizes
        self.coregs = coregs
        self.diags = (
            [np.ones(len(sizes)) for _ in coregs]
            if diags is None else diags)

        self.xss = [np.random.rand(sz) for sz in sizes]
        self.yss = [np.random.rand(sz) for sz in sizes]

    def gen_lmc(self, m):
        lmc = LMC(self.xss, self.yss, normalize=False, kernels=self.kerns, m=m)
        for lmc_coreg, coreg in zip(lmc.coreg_vecs, self.coregs):
            lmc_coreg[:] = coreg
        for lmc_coreg, coreg in zip(lmc.coreg_diags, self.diags):
            lmc_coreg[:] = coreg
        lmc.noise[:] = np.ones(len(self.sizes))
        return lmc

    def gen_exact_mat(self):
        exact_mat = np.identity(sum(self.sizes))
        for coreg, diag, kern in zip(self.coregs, self.diags, self.kerns):
            coreg_mat = np.outer(coreg, coreg).astype(float)
            coreg_mat += np.diag(diag)
            exact_mat += np.bmat(
                [[coreg_mat[i, j] *
                  self.pairwise_dists(kern, self.xss[i], self.xss[j])
                  for j in range(len(self.sizes))]
                 for i in range(len(self.sizes))]).A
        return exact_mat

    @staticmethod
    def pairwise_dists(kern, xs1, xs2):
        dists = scipy.spatial.distance.cdist(
            xs1.reshape(-1, 1), xs2.reshape(-1, 1))
        return kern.from_dist(dists).reshape(len(xs1), len(xs2))

class LMCTest(RandomTest):

    def setUp(self):
        super().setUp()

    @staticmethod
    def case_1d():
        kerns = [RBF(inv_lengthscale=3)]
        szs = [15]
        coregs = [[1]]
        return ExactAnalogue(kerns, szs, coregs)

    @staticmethod
    def case_2d():
        kerns = [RBF(inv_lengthscale=3),
                 RBF(inv_lengthscale=2)]
        szs = [15, 20]
        coregs = [[1, 2], [3, 4]]
        return ExactAnalogue(kerns, szs, coregs)

    @staticmethod
    def case_large():
        kerns = [RBF(inv_lengthscale=3),
                 RBF(inv_lengthscale=2),
                 RBF(inv_lengthscale=1)]
        szs = [15, 20, 10, 12, 13]
        coregs = [[1, 1, 1, 1, 2], [2, 1, 2, 1, 2], [-1, 1, -1, -1, -1]]
        return ExactAnalogue(kerns, szs, coregs)

    @staticmethod
    def avg_entry_diff(x1, x2):
        return np.fabs(x1 - x2).mean()

    def check_kernel_reconstruction(self, exact):
        reconstruct = lambda x: x.kernel['apprx'].ski.as_numpy()
        actual = reconstruct(exact.gen_lmc(sum(exact.sizes)))
        exact_mat = exact.gen_exact_mat()
        tol = 1e-4
        np.testing.assert_allclose(
            exact_mat, actual, rtol=tol, atol=tol)
        avg_diff_sz = self.avg_entry_diff(exact_mat, actual)

        actual = reconstruct(exact.gen_lmc(sum(exact.sizes) * 2))
        np.testing.assert_allclose(
            exact_mat, actual, rtol=tol, atol=tol)
        avg_diff_2sz = self.avg_entry_diff(exact_mat, actual)

        self.assertGreater(avg_diff_sz, avg_diff_2sz)

    def check_normal_quadratic(self, exact):
        exact_mat = exact.gen_exact_mat()
        y = np.hstack(exact.yss)
        Kinv_y = np.linalg.solve(exact_mat, y)
        expected = y.dot(Kinv_y)

        lmc = exact.gen_lmc(sum(exact.sizes))
        lmc.TOL = 1e-15 # tighten tolerance for tests
        tol = 1e-4

        actual = lmc.normal_quadratic()
        np.testing.assert_allclose(expected, actual, rtol=tol, atol=tol)

    def check_fit(self, ea, noise_sd, true_func): # pylint: disable=too-many-locals
        assert all(x <= 0.1 for x in noise_sd)
        assert len(noise_sd) == len(true_func)
        noises = [np.random.randn(len(xs)) * sd
                  for xs, sd in zip(ea.xss, noise_sd)]
        ea.yss = [f(xs) + noise
                  for f, xs, noise in zip(true_func, ea.xss, noises)]
        lmc = ea.gen_lmc(sum(ea.sizes))

        mu, var = lmc.predict(ea.xss)
        avg_err_before = [np.fabs(m - ys).mean() for m, ys in zip(mu, ea.yss)]
        avg_var_before = [v.mean() for v in var]
        ll_before = lmc.log_likelihood()
        lmc.optimize(optimizer='lbfgs')
        mu, var = lmc.predict(ea.xss)
        err_after = [np.fabs(m - ys) for m, ys in zip(mu, ea.yss)]
        avg_err_after = [err.mean() for err in err_after]
        avg_var_after = [v.mean() for v in var]
        ll_after = lmc.log_likelihood()

        for before, after in zip(avg_err_before, avg_err_after):
            self.assertGreater(before, after)

        for before, after in zip(avg_var_before, avg_var_after):
            self.assertGreater(before, after)

            self.assertGreater(ll_after, ll_before)

        # Probibalistic but very, very likely to hold bounds
        # These will only catch gross errors

        for errs, output_vars in zip(err_after, var):
            sds = np.sqrt(output_vars)
            nabove_3sig = np.count_nonzero(errs > 3 * sds)
            # Note 5% is two sigma, intentionally.
            self.assertGreater(0.05, nabove_3sig / len(errs))

        # Be within a magnitude of the noise sd
        for avg_var, sd in zip(avg_var_after, noise_sd):
            actual_sd = np.sqrt(avg_var)
            self.assertGreater(actual_sd, sd / 10)
            # self.assertGreater(sd * 10, actual_sd)

        # TODO better verification necessary, as soon as we get better
        # optimization.

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
        self.check_fit(ea, noise_sd, true_func)

    # TODO(cleanup): introduce testing for exact analogue, compare
    #                side-by-side.

    # TODO(fix): fix broken unit tests

    @unittest.skip('broken')
    def test_2d_fit(self):
        ea = self.case_2d()
        noise_sd = [0.05, 0.08]
        true_func = [np.sin, np.cos]
        self.check_fit(ea, noise_sd, true_func)

    @unittest.skip('broken')
    def test_2d_fit_noisediff(self):
        ea = self.case_2d()
        noise_sd = [1e-8, 0.09]
        true_func = [np.sin, np.cos]
        self.check_fit(ea, noise_sd, true_func)

    @unittest.skip('broken')
    def test_2d_1k_fit_large_offset(self):
        kerns = [RBF(inv_lengthscale=3)]
        szs = [30, 40]
        coregs = [[1, 1]]
        ea = ExactAnalogue(kerns, szs, coregs)
        noise_sd = [0.02, 0.08]
        true_func = [np.sin, lambda x: np.cos(x) + 100]
        self.check_fit(ea, noise_sd, true_func)
