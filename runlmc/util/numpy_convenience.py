# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

"""
Convenience functions for working with numpy arrays.
"""

from itertools import accumulate

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

def tesselate(nparr, lenit):
    """
    Create a ragged array by splitting `nparr` into contiguous
    segments of size determined by the length list `lenit`

    :param nparr: array to split along axis 0.
    :param lenit: iterator of lengths to split into.
    :return: A list of size equal to `lenit`'s iteration with `nparr`'s
             segments split into corresponding size chunks.
    :raise ValueError: if the sum of lengths doesn't correspond to the array
                       size.
    """
    ends = np.fromiter(accumulate(lenit), dtype=np.int)
    if len(ends) == 0:
        raise ValueError('no segment lengths specified')
    if nparr.shape[0] != ends[-1]:
        raise ValueError('shape {}[0] != {} num elements'.format(
            nparr.shape, ends[-1]))
    return np.split(nparr, ends[:-1])
