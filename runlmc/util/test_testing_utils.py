# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import os

import numpy as np
import scipy.linalg

from . import testing_utils as utils

class RandomTestTest(utils.RandomTest):
    def setUp(self):
        oldenv = os.getenv('SEED')
        os.environ['SEED'] = '123456'
        super().setUp()
        if oldenv is None:
            os.unsetenv('SEED')
        else:
            os.environ['SEED'] = oldenv

    def test_random(self):
        self.assertEqual(self.seed, 123456)

class PSDConstructionTest(utils.RandomTest):

    SIZES = [1 << n for n in range(2, 6)]

    def generate(self, f):
        return map(f, self.SIZES)

    def generate_toep(self, f):
        return map(scipy.linalg.toeplitz, self.generate(f))

    def test_poor_cond_toep_distinct(self):
        self.assertFalse(np.allclose(
            utils.poor_cond_toep(50), utils.poor_cond_toep(50)))

    def test_poor_cond_toep_psd(self):
        mats = list(self.generate_toep(utils.poor_cond_toep))
        eigss = [np.linalg.eigvalsh(mat)
                 for mat in mats]
        for eigs, mat in zip(eigss, mats):
            msg = '\nmatrix\n{}\neigs\n{}'.format(mat, eigs)
            self.assertTrue(np.all(eigs >= 0), msg=msg)

    def test_poor_cond_toep_ill_conditioned(self):
        logs = np.log([np.linalg.cond(mat)
                       for mat in self.generate_toep(utils.poor_cond_toep)])
        growth = logs / np.roll(logs, 1)
        np.testing.assert_array_less(1.1 * np.ones(len(logs) - 1), growth[1:])

    def test_random_toep_distinct(self):
        self.assertFalse(np.allclose(
            utils.random_toep(50), utils.random_toep(50)))

    def test_random_toep_psd(self):
        mats = list(self.generate_toep(utils.random_toep))
        eigss = [np.linalg.eigvalsh(mat)
                 for mat in mats]
        for eigs, mat in zip(eigss, mats):
            msg = '\nmatrix\n{}\neigs\n{}'.format(mat, eigs)
            self.assertTrue(np.all(eigs >= 0), msg=msg)

    def test_exp_decr_toep_distinct(self):
        self.assertFalse(np.allclose(
            utils.exp_decr_toep(50), utils.exp_decr_toep(50)))

    def test_exp_decr_toep_psd(self):
        mats = list(self.generate_toep(utils.exp_decr_toep))
        eigss = [np.linalg.eigvalsh(mat)
                 for mat in mats]
        for eigs, mat in zip(eigss, mats):
            msg = '\nmatrix\n{}\neigs\n{}'.format(mat, eigs)
            self.assertTrue(np.all(eigs >= 0), msg=msg)

    def test_rand_pd_distinct(self):
        self.assertFalse(np.allclose(
            utils.rand_pd(50), utils.rand_pd(50)))

    def test_rand_pd_symm(self):
        for mat in self.generate(utils.rand_pd):
            msg = '\nmatrix\n{}'.format(mat)
            np.testing.assert_allclose(mat, mat.T, err_msg=msg)

    def test_rand_pd_psd(self):
        mats = list(self.generate(utils.rand_pd))
        eigss = [np.linalg.eigvalsh(mat)
                 for mat in mats]
        for eigs, mat in zip(eigss, mats):
            msg = '\nmatrix\n{}\neigs\n{}'.format(mat, eigs)
            self.assertTrue(np.all(eigs >= 0), msg=msg)
