# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

"""
Convenience functions for working with numpy arrays.
"""

from itertools import accumulate

import numpy as np
import scipy.linalg
import scipy.sparse.linalg


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
    if not ends.size:
        raise ValueError('no segment lengths specified')
    if nparr.shape[0] != ends[-1]:
        raise ValueError('shape {}[0] != {} num elements'.format(
            nparr.shape, ends[-1]))
    return np.split(nparr, ends[:-1])


EPS = np.finfo('float64').eps


def search_descending(x, xs, inclusive):
    """
    :param x: threshold
    :param xs: descending-ordered array to search
    :param inclusive: whether to include values of `x` in `xs`
    :returns: the largest index index `i` such that `xs[:i] >= x`
              if `inclusive` else `xs[:i] > x`.
    :raises ValueError: if array is not weakly decreasing
    """
    xs = np.array(xs)
    if np.any(np.diff(xs) > 0):
        raise ValueError('array is not weakly decreasing:\n{}'.format(xs))
    option = 'left' if inclusive else 'right'
    idx = np.searchsorted(xs[::-1], x, option)
    return len(xs) - idx


def smallest_eig(top):
    """
    :param top: top row of Toeplitz matrix to get eigenvalues for
    :type top numpy.ndarray:
    :returns: the smallest eigenvalue
    """

    if len(top) == 1:
        return top[0]

    return scipy.linalg.eigvalsh(scipy.linalg.toeplitz(top)).min()

# TODO(test)


def symm_2d_list_map(f, arr, D, dtype='object'):
    """Symmetric map construction"""
    out = np.empty((D, D), dtype=dtype)
    for i in range(D):
        for j in range(i, D):
            out[i, j] = f(arr[i, j])
            out[j, i] = out[i, j]
    return out

# TODO(test)


def begin_end_indices(lens):
    ends = np.add.accumulate(lens)
    begins = np.roll(ends, 1)
    begins[0] = 0
    return begins, ends
