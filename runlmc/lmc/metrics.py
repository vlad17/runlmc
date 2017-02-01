# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

class Metrics:
    def __init__(self):
        self.iterations = []
        self.grad_norms = []
        self.grad_error = []
        self.solv_error = []
        self.log_likely = []
