# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

from ..linalg.composition import Composition
from ..linalg.matrix import Matrix

# TODO(test)
class SKI(Composition):
    def __init__(self, K, W, WT):
        self.W = W
        self.K = K
        self.WT = WT
        super().__init__([
            Matrix.wrap(W.shape, W.dot),
            K,
            Matrix.wrap(WT.shape, WT.dot)])

    def as_numpy(self):
        WKT = self.W.dot(self.K.as_numpy().T)
        return self.W.dot(WKT.T)

    def upper_eig_bound(self):
        return self.K.upper_eig_bound() * self.shape[0] / self.m
