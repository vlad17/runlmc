# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

"""
Convenience functions for working with numpy arrays.
"""

import numpy as np

def map_entries(f, nparr):
    """
    Map a function over a numpy array.

    :param f: single-parameter function over the same types
    :param `np.ndarray` nparr: arbitrary numpy array
    :return: A numpy array with `f` evaluated on each
             element of the same shape.
    """
    if nparr.size == 0:
        return nparr
    it = np.nditer(nparr)
    shape = nparr.shape
    dtype = nparr.dtype
    return np.fromiter(map(f, it), dtype).reshape(shape)
