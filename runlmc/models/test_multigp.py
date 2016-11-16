# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import unittest

class MultiGPTest(unittest.TestCase):
    # TODO: add tests for the following functionality (may need a dummy
    # derived class with its own priored param, mean, and _raw_predict):
    #   - name
    #   - mean function ValueError on construction
    #   - Xs/Ys error if length mismatch on contruction
    #   - No normalization if no normalizer
    #   - Normalization if there is a normalizer (for normalizer itself,
    #     make sure it handles all-equal case, warns if low sd)
    #   - Make sure dummy param, mean function perpetuate priors
    #   - Make sure dummy param, mean function are notified of gradient
    #   - predict_quantiles works property for >2 tups
    pass
