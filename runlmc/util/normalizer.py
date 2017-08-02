# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

# This file was modified from the GPy project. Its file header is replicated
# below. Its LICENSE.txt is replicated in the LICENSE file for this directory.

# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np


class Norm(object):
    def __init__(self):
        self.mean = None
        self.std = None

    def scale_by(self, Y):
        """
        Use data matrix Y as normalization space to work in.
        """
        Y = np.ma.masked_invalid(Y, copy=False)
        self.mean = Y.mean(0).view(np.ndarray)
        self.std = Y.std(0).view(np.ndarray)

    def normalize(self, Y):
        """
        Project Y into normalized space
        """
        if not self.scaled():
            raise AttributeError(
                'Norm object not initialized yet,'
                'try calling scale_by(data) first.')
        return (Y - self.mean) / self.std

    def inverse_mean(self, X):
        """
        Project the normalized object X into space of Y
        """
        return (X * self.std) + self.mean

    def inverse_variance(self, var):
        return var * (self.std**2)

    def scaled(self):
        """
        Whether this Norm object has been initialized.
        """
        return self.mean is not None
