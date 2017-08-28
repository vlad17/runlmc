# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import numpy as np


class ParameterValues:
    def __init__(self, coreg_vecs, coreg_diags, kernels, lens, y, noise,
                 nkernels=None):
        if nkernels is None:
            nkernels = {'lmc': len(kernels), 'slfm': 0, 'indep': 0}

        self.coreg_vecs = coreg_vecs
        self.coreg_diags = coreg_diags

        self.kernels = kernels
        self.nkernels = nkernels

        self.coreg_mats = [a.T.dot(a) + np.diag(k)
                           for a, k in zip(coreg_vecs, self.coreg_diags)]

        self.lens = lens
        self.noise = noise
        self.y = y
        self.n = len(self.y)
        self.D = len(self.noise)
        self.Q = len(self.kernels)

    @staticmethod
    def generate(lmc_model):
        return ParameterValues(
            [x.values for x in lmc_model.coreg_vecs],
            [x.values for x in lmc_model.coreg_diags],
            lmc_model.kernels,
            list(map(len, lmc_model.Xs)),
            lmc_model.y,
            lmc_model.noise.values,
            lmc_model.nkernels)
