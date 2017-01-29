# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

from ..linalg.symmetric_matrix import SymmetricMatrix

class SKI(SymmetricMatrix):
    def __init__(self, K, W, WT):
        super().__init__(W.shape[0])
        self.m = W.shape[1]
        self.K = K
        self.W = W
        self.WT = WT

    def as_numpy(self):
        WKT = self.W.dot(self.K.as_numpy().T)
        return self.W.dot(WKT.T)

    def matvec(self, x):
        return self.W.dot(self.K.matvec(self.WT.dot(x)))

    def upper_eig_bound(self):
        return self.K.upper_eig_bound() * self.shape[0] / self.m
