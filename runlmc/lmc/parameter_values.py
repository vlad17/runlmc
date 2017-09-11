# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import numpy as np

# TODO(cleanup): elim parametervalues thanks to functional_kernel OO
# get rid of _functional_kernel. accesses too


class ParameterValues:
    def __init__(self, functional_kernel, lens, y):

        self.functional_kernel = functional_kernel
        self.kernels = functional_kernel._kernels
        self.nkernels = functional_kernel._nkernels
        self.coreg_vecs = self.functional_kernel._coreg_vecs
        self.coreg_diags = self.functional_kernel._coreg_diags
        self.coreg_mats = [a.T.dot(a) + np.diag(k)
                           for a, k in zip(self.coreg_vecs, self.coreg_diags)]

        self.lens = lens
        self.noise = self.functional_kernel._noise
        self.y = y
        self.n = len(self.y)
        self.D = len(self.noise)
        self.Q = len(self.functional_kernel._kernels)

    @staticmethod
    def generate(lmc_model):
        return ParameterValues(
            lmc_model._functional_kernel,
            list(map(len, lmc_model.Xs)),
            lmc_model.y)
