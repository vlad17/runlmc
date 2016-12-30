# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import numpy as np
import scipy.linalg
import scipy.spatial.distance

from .lmc import LMC
from ..kern.rbf import RBF
from ..util.testing_utils import RandomTest

class LMCTest(RandomTest):

    def setUp(self):
        super().setUp()

    @staticmethod
    def pairwise_dists(kern, xs1, xs2):
        dists = scipy.spatial.distance.cdist(
            xs1.reshape(-1, 1), xs2.reshape(-1, 1))
        return kern.from_dist(dists).reshape(len(xs1), len(xs2))

    @staticmethod
    def avg_entry_diff(x1, x2):
        return np.fabs(x1 - x2).mean()


    def check_kernel_reconstruction(self, kerns, szs, coregs):
        assert len(coregs) == len(kerns)
        assert set(map(len, coregs)) == {len(szs)}

        tol = LMC.TOL

        def gen_lmc(m):
            lmc = LMC(xss, yss, normalize=False, kernels=kerns, m=m)
            for lmc_coreg, coreg in zip(lmc.coreg_vecs, coregs):
                lmc_coreg[:] = coreg
            lmc.noise[:] = np.ones(len(szs))
            return lmc

        xss = [np.random.rand(sz) for sz in szs]
        yss = [np.random.rand(sz) for sz in szs]

        expected = np.identity(sum(szs))
        for coreg, kern in zip(coregs, kerns):
            coreg_mat = np.outer(coreg, coreg)
            expected += np.bmat([
                [coreg_mat[i, j] * self.pairwise_dists(kern, xss[i], xss[j])
                 for j in range(len(szs))]
                for i in range(len(szs))]).A

        actual = gen_lmc(sum(szs)).K_SKI()
        np.testing.assert_allclose(expected, actual, rtol=tol, atol=tol)
        avg_diff_sz = self.avg_entry_diff(expected, actual)

        actual = gen_lmc(sum(szs) * 2).K_SKI()
        np.testing.assert_allclose(expected, actual, rtol=tol, atol=tol)
        avg_diff_2sz = self.avg_entry_diff(expected, actual)

        self.assertGreater(avg_diff_sz, avg_diff_2sz)

    def test_no_kernel(self):
        mapnp = lambda x: list(map(np.array, x))
        basic_Xs = mapnp([[0, 1, 2], [0.5, 1.5, 2.5]])
        basic_Ys = mapnp([[5, 6, 7], [7, 6, 5]])
        self.assertRaises(ValueError, LMC,
                          basic_Xs, basic_Ys, kernels=[])

    def test_kernel_reconstruction_1d(self):
        kerns = [RBF(variance=2, inv_lengthscale=3)]
        szs = [15]
        coregs = [[1]]
        self.check_kernel_reconstruction(kerns, szs, coregs)

    def test_kernel_reconstruction_2d(self):
        kerns = [RBF(variance=2, inv_lengthscale=3),
                 RBF(variance=3, inv_lengthscale=2)]
        szs = [15, 20]
        coregs = [[1, 2], [3, 4]]
        self.check_kernel_reconstruction(kerns, szs, coregs)

    def test_kernel_reconstruction_large(self):
        kerns = [RBF(variance=2, inv_lengthscale=3),
                 RBF(variance=3, inv_lengthscale=2),
                 RBF(variance=1, inv_lengthscale=1)]
        szs = [15, 20, 10, 12, 13]
        coregs = [[1, 1, 1, 1, 2], [2, 1, 2, 1, 2], [-1, 1, -1, -1, -1]]
        self.check_kernel_reconstruction(kerns, szs, coregs)


    def test_normal_quadratic(self):
        # should agree with K_SKI (make a randomtest?)
        pass

    def test_optimization(self):
        # on a random example, NLL should decrease before/after.
        pass

    def test_1d_fit(self):
        # internally, fit should get within noise on training data
        # up to tolerance (use sin).
        pass

    def test_2d_fit_nocov(self):
        # internally, fit should get within noise on training data up
        # to tolerance (use sin/cos)
        # note coregionalization may be found even if not there - nonconvex
        # problem won't necessarily find the right solution -> don't check
        # params, just fit
        pass

    # TODO: test_2d_fit_cov - with covariance
    # TODO: test_coreg_nocov - requires rank 2, single kernel, l1 prior
    #       on coreg (should find identity matrix approx coregionalization
    #       after a couple random restarts (choose l2 err minimizing one,
    #       on data with no noise).
    # TODO: same as above, but find the covariance. (check we don't go to the
    #       identity)
