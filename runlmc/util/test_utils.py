# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import os
import random
import time

import numpy as np

# I wrote a similar class in databricks/spark-sklearn, but the task is
# small and common enough that the code is basically the same.
class RandomTest:
    """
    This test case mixin sets the random seed to be based on the time
    that the test is run.

    If there is a `SEED` variable in the enviornment, then this is used as the
    seed.

    Sets both random and numpy.random.
    Prints the seed to stdout before running each test case.
    """

    def setUp(self):
        seed = os.getenv("SEED")
        seed = np.uint32(seed if seed else time.time())

        print('Random test using SEED={}'.format(seed))

        random.seed(seed)
        np.random.seed(seed)
