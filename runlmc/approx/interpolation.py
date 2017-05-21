# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

# The interp_cubic code below was modified from MSGP demo code provided by
# Wilson. The demo code itself is based off of extending components
# in GPML, which is BSD 2-clause licenced. I've replicated the
# copyrights and/or licences to both code source in this repository's
# LICENSE file.

import logging

import numpy as np
import scipy
import scipy.sparse

from ..util.numpy_convenience import begin_end_indices

_LOG = logging.getLogger(__name__)

def cubic_kernel(x):
    """
    The cubic convolution kernel can be used to compute interpolation
    coefficients. Its definition is taken from:

    Cubic Convolution Interpolation for Digital Image Processing by
    Robert G. Keys.

    It is supported on values of absolute magnitude less than
    2 and is defined as:

    .. math::
        \\newcommand{abs}[1]{\\left\\vert{#1}\\right\\vert}
        u(x) = \\begin{cases}
        \\frac{3}{2}\\abs{x}^3-\\frac{5}{2}\\abs{x}^2+1 & 0\\le \\abs{x}\\le 1
        \\\\
        \\frac{3}{2}\\abs{x}^3+\\frac{5}{2}-4\\abs{x}+2 & 1 < \\abs{x} \\le 2
        \\end{cases}

    :param x: input array
    :type x: :class:`numpy.ndarray`
    :returns: :math:`u` vectorized over `x`
    """
    y = np.zeros_like(x)
    x = np.fabs(x)
    if np.any(x > 2):
        raise ValueError('only absolute values <= 2 allowed')
    q = x <= 1
    y[q] = ((1.5 * x[q] - 2.5) * x[q]) * x[q] + 1
    q = ~q
    y[q] = ((-0.5 * x[q] + 2.5) * x[q] - 4) * x[q] + 2
    return y

def interp_cubic(grid, samples):
    """
    Given a one dimensional grid `grid` of size `m` (that's sorted) and
    `n` sample points `samples` contained in `grid[1:-1]`, compute the
    interpolation coefficients for a cubic interpolation on the grid.

    An interpolation coefficient matrix `M` is then an `n` by `m` matrix
    that has 4 entries per row.

    For a (vectorized) twice-differentiable function `f`, `M.dot(f(grid))`
    approaches `f(sample)` at a rate of :math:`O(m^{-3})`.

    `samples` should be contained with the range of `grid`, but this method
    will make do will work with what it has (it will clip things back into
    range).

    :returns: the interpolation coefficient matrix
    :raises ValueError: if any of the following hold true:

        #. `grid` or `samples` are not 1-dimensional
        #. `grid` size less than 4
        #. `grid` is not equispaced
    """

    grid_size = len(grid)
    n_samples = samples.size

    if n_samples == 0:
        return scipy.sparse.csr_matrix((0, grid_size), dtype=float)

    if grid.ndim != 1:
        raise ValueError('grid dim {} should be 1'.format(grid.ndim))

    if samples.ndim != 1:
        raise ValueError('samples dim {} should be 1'.format(samples.ndim))

    if grid_size < 4:
        raise ValueError('grid size {} must be >=4'.format(grid_size))

    if samples.min() <= grid[0] or samples.max() >= grid[-1]:
        _LOG.warning('range of samples [%f, %f] outside grid range [%f, %f]',
                     samples.min(), samples.max(), grid[0], grid[-1])

    delta = grid[1] - grid[0]
    factors = (samples - grid[0]) / delta
    # closest refers to the closest grid point that is smaller
    idx_of_closest = np.floor(factors)
    dist_to_closest = factors - idx_of_closest # in units of delta

    csr = scipy.sparse.csr_matrix((n_samples, grid_size), dtype=float)
    for conv_idx in range(-2, 2): # cubic conv window
        coeff_idx = idx_of_closest - conv_idx
        coeff_idx[coeff_idx < 0] = 0 # threshold (no wraparound below)
        coeff_idx[coeff_idx >= grid_size] = grid_size - 1 # none above

        relative_dist = dist_to_closest + conv_idx
        data = cubic_kernel(relative_dist)
        col_idx = coeff_idx
        ind_ptr = np.arange(0, n_samples + 1)
        csr += scipy.sparse.csr_matrix((data, col_idx, ind_ptr),
                                       shape=(n_samples, grid_size))
    return csr

# TODO(test)
# TODO(cleanup) - refactor to get rid of pylint warning
def multi_interpolant(Xs, inducing_grid): # pylint: disable=too-many-locals
    """
    Creates a sparse CSR matrix across multiple inputs `Xs`.

    Each input is mapped onto the inducing grid with a cubic interpolation,
    with :func:`runlmc.approx.interpolation.interp_cubic`.

    This induces :math:`n_i\\times m` interpolation matrices :math:`W_i` for
    the :math:`i`-th element of `Xs` onto the shared inducing grid.

    :param Xs: list of 1-dimensional numpy vectors, the inputs.
    :param inducing_gird: 1-dimensional vector of grid points
    :return: the rectangular block diagonal matrix of :math:`W_i`.
    """
    multiout_grid_sizes = np.arange(len(Xs)) * len(inducing_grid)
    Ws = [interp_cubic(inducing_grid, X) for X in Xs]

    row_lens = [len(X) for X in Xs]
    row_begins, row_ends = begin_end_indices(row_lens)
    order = row_ends[-1]

    col_lens = [W.nnz for W in Ws]
    col_begins, col_ends = begin_end_indices(col_lens)
    width = col_ends[-1]

    ind_starts = np.roll(np.add.accumulate([W.indptr[-1] for W in Ws]), 1)
    ind_starts[0] = 0
    ind_ptr = np.append(np.repeat(ind_starts, row_lens), width)
    data = np.empty(width)
    col_indices = np.repeat(multiout_grid_sizes, col_lens)
    for rbegin, rend, cbegin, cend, W in zip(
            row_begins, row_ends, col_begins, col_ends, Ws):
        ind_ptr[rbegin:rend] += W.indptr[:-1]
        data[cbegin:cend] = W.data
        col_indices[cbegin:cend] += W.indices

    ncols = len(Xs) * len(inducing_grid)
    return scipy.sparse.csr_matrix(
        (data, col_indices, ind_ptr), shape=(order, ncols))
