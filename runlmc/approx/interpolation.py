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
        u(x) = \\begin{cases}
        \\frac{3}{2}\\left\\vert{x}\\right\\vert^3-\\frac{5}{2}\\left\\vert{x}
        \\right\\vert^2+1 & 0\\le \\left\\vert{x}\\right\\vert\\le 1
        \\\\
        \\frac{3}{2}\\left\\vert{x}\\right\\vert^3+\\frac{5}{2}-4\\left
        \\vert{x}\\right\\vert+2 & 1 < \\left\\vert{x}\\right\\vert \\le 2
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
    `n` sample points `samples`, compute the
    interpolation coefficients for a cubic interpolation on the grid.

    An interpolation coefficient matrix `M` is then an `n` by `m` matrix
    that has 4 entries per row.

    For a (vectorized) twice-differentiable function `f`, `M.dot(f(grid))`
    approaches `f(sample)` at a rate of :math:`O(m^{-3})`.

    `samples` should be contained with the range of `grid`, but this method
    will create a matrix capable of handling extrapolation.

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
    dist_to_closest = factors - idx_of_closest  # in units of delta

    csr = scipy.sparse.csr_matrix((n_samples, grid_size), dtype=float)
    for conv_idx in range(-2, 2):  # cubic conv window
        coeff_idx = idx_of_closest - conv_idx
        coeff_idx[coeff_idx < 0] = 0  # threshold (no wraparound below)
        coeff_idx[coeff_idx >= grid_size] = grid_size - 1  # none above

        relative_dist = dist_to_closest + conv_idx
        data = cubic_kernel(relative_dist)
        col_idx = coeff_idx
        ind_ptr = np.arange(n_samples + 1)
        csr += scipy.sparse.csr_matrix((data, col_idx, ind_ptr),
                                       shape=(n_samples, grid_size))
    return csr


def multi_interpolant(Xs, *inducing_grids):  # pylint: disable=too-many-locals
    """
    Creates a sparse CSR matrix interpolant across multiple inputs `Xs`.

    Each input is mapped onto the inducing grid with a cubic interpolation,
    with :func:`runlmc.approx.interpolation.interp_cubic` or
    :func:`runlmc.approx.interpolation.interp_bicubic`, depending on the
    dimensionality of `Xs`.

    This induces :math:`n_i\\times m` interpolation matrices :math:`W_i` for
    the :math:`i`-th element of `Xs` onto the inducing grid, which is
    shared between all `Xs`. Note that :math:`m` is the total size of
    the Cartesian product of the inducing grids for each dimension of the
    input. This also implies that the number of inducing grid axes passed
    as arguments to `multi_interpolant` must be equal to the dimension of
    `Xs`.

    :param Xs: list of :math:`n_i`-by-:math:`d` input points, where
        :math:`d` may be either 1 or 2. In the case :math:`d=1` the input
        matrices can be vectors.
    :param inducing_grids: list 1-dimensional vector of grid points, one for
        each input dimension
    :return: the rectangular block diagonal matrix of :math:`W_i`.
    """
    m = np.prod([len(grid) for grid in inducing_grids])
    multiout_grid_sizes = np.arange(len(Xs)) * m

    if Xs[0].ndim == 1 or Xs[0].shape[1] == 1:
        Ws = [interp_cubic(inducing_grids[0], X.ravel()) for X in Xs]
    else:
        gridx = inducing_grids[0]
        gridy = inducing_grids[1]
        Ws = [interp_bicubic(gridx, gridy, X) for X in Xs]

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
    for rows, cols, W in zip(
            map(slice, row_begins, row_ends),
            map(slice, col_begins, col_ends),
            Ws):
        ind_ptr[rows] += W.indptr[:-1]
        data[cols] = W.data
        col_indices[cols] += W.indices

    ncols = len(Xs) * m
    return scipy.sparse.csr_matrix(
        (data, col_indices, ind_ptr), shape=(order, ncols))


def autogrid(Xs, lo, hi, m):
    """
    Generate a grid from `lo` to `hi` with `m` points, but with sensible
    defaults based on `Xs` if any of the other parameters are `None`.

    In particular, the defaults are chosen such that all elements in `Xs`
    fall within the grid by two elements on both sides. If a user's values
    do not guarantee this property, they are changed.

    :param Xs: list of 1-dimensional numpy vectors, the inputs.
    :param lo: optional lower bound
    :param hi: optional upper bound
    :param m: optional grid size
    :return: a pair of equally-spaced points between low, hi
    """

    m = m or sum(len(X) for X in Xs) // len(Xs)
    max_lo = min(X.min() for X in Xs)
    min_hi = max(X.max() for X in Xs)

    lo = min(lo, max_lo) if lo else max_lo
    hi = max(hi, min_hi) if hi else min_hi

    delta = (hi - lo) / m
    lo -= 2 * delta
    hi += 2 * delta
    m += 4

    return np.linspace(lo, hi, m), m


def interp_bicubic(gridx, gridy, samples):  # pylint: disable=too-many-locals
    """
    Given an implicit two dimensional grid from the Cartesian product of
    `gridx` and `gridy`, with sizes `m1,m2`, respectively (such that each
    grid is an equispaced increasing sequence of values), and given
    `n` sample points (which should be 2D), this computes
    interpolation coefficients for a cubic interpolation on the grid.

    An interpolation coefficient matrix `M` is then an `n` by `m1*m2` matrix
    that has 16 entries per row.

    For a (vectorized) twice-differentiable function `f`,
    `M.dot(f(cartesian_product(gridx, gridy)).ravel())`
    approaches `f(sample)` at a rate of :math:`O(m^{-3})`.

    :returns: the interpolation coefficient matrix
    :raises ValueError: if any conditions similar to those in
        :meth:`interp_cubic` are violated.
    """

    mx, my = gridx.size, gridy.size
    n = samples.shape[0]

    if n == 0:
        return scipy.sparse.csr_matrix((0, mx * my), dtype=float)

    for s in ['gridx', 'gridy']:
        grid = eval(s)  # pylint: disable=eval-used
        if grid.ndim != 1:
            raise ValueError('{} dim {} should be 1'.format(s, grid.ndim))
        if grid.size < 4:
            raise ValueError('grid size {} must be >=4'.format(grid.size))

    if samples.ndim != 2 or samples.shape[1] != 2:
        raise ValueError(
            'expecting 2d samples, got shape {}'.format(samples.shape))

    if samples[:, 0].min() <= gridx[0] or samples[:, 0].max() >= gridx[-1]:
        _LOG.warning('x range of samples [%f, %f] outside grid range [%f, %f]',
                     samples[:, 0].min(), samples[:, 0].max(),
                     gridx[0], gridx[-1])

    if samples[:, 1].min() <= gridy[0] or samples[:, 1].max() >= gridy[-1]:
        _LOG.warning('y range of samples [%f, %f] outside grid range [%f, %f]',
                     samples[:, 1].min(), samples[:, 1].max(),
                     gridy[0], gridy[-1])

    dx, dy = gridx[1] - gridx[0], gridy[1] - gridy[0]

    # For each sample point (sx, sy), first, generate virtual sample points
    # (sx, gridy(-2)), (sx, gridy(-1)), (sx, gridy(0)), (sx, gridy(1))
    # where gridy(range(-2,2)) corresponds to the four-point bubble of grid
    # points in gridy around sy.
    #
    # Next, use regular cubic interpolation to generate an interpolated
    # function value for all of the four above points: for i in range(-2, 2),
    # interpolate the function value at (sx, gridy(i)) against the full
    # mesh.
    #
    # Finish by interpolating the interpolated values along the y dimension.

    factors_y = (samples[:, 1] - gridy[0]) / dy
    idx_of_closest_y = np.floor(factors_y)
    dist_to_closest_y = factors_y - idx_of_closest_y

    factors_x = (samples[:, 0] - gridx[0]) / dx
    idx_of_closest_x = np.floor(factors_x)
    dist_to_closest_x = factors_x - idx_of_closest_x

    xcsrs = []
    ycsr = scipy.sparse.csr_matrix((n, n * 4), dtype=float)
    for yconv_idx in range(-2, 2):
        ycoeff_idx = idx_of_closest_y - yconv_idx
        ycoeff_idx[ycoeff_idx < 0] = 0  # threshold (no wraparound below)
        ycoeff_idx[ycoeff_idx >= my] = my - 1  # none above

        # vector of (sx, gridy(i)) is just
        # Ui = np.column_stack([samples[:, 1], gridy[coeff_idx]])
        #
        # We find a matrix Mi such that
        # f(Ui) = Mi.f(gridx x gridy)
        # recall for each (sx, gridy(i)) we interpolate the x-dimension
        xcsr = scipy.sparse.csr_matrix((n, mx * my), dtype=float)
        for xconv_idx in range(-2, 2):
            xcoeff_idx = idx_of_closest_x - xconv_idx
            xcoeff_idx[xcoeff_idx < 0] = 0  # threshold (no wraparound below)
            xcoeff_idx[xcoeff_idx >= mx] = mx - 1  # none above

            xrelative_dist = dist_to_closest_x + xconv_idx
            xdata = cubic_kernel(xrelative_dist)
            # index into appropriate x value of f(gridx x gridy)
            xcol_idx = xcoeff_idx * my + ycoeff_idx
            xind_ptr = np.arange(n + 1)
            xcsr += scipy.sparse.csr_matrix((xdata, xcol_idx, xind_ptr),
                                            shape=(n, mx * my))
        xcsrs.append(xcsr)

        # for every fixed sx = samples[j, 0] we'd like to
        # interpolate sy from Ui[j] at all i in range(-2, 2)
        # at the end of this loop we don't have all Ui ready, but
        # we can still get the interpolation coefficients for the current
        # i == yconv_idx
        yrelative_dist = dist_to_closest_y + yconv_idx
        ydata = cubic_kernel(yrelative_dist)
        ycol_idx = np.arange(n) + n * (yconv_idx + 2)
        yind_ptr = np.arange(n + 1)
        ycsr += scipy.sparse.csr_matrix((ydata, ycol_idx, yind_ptr),
                                        shape=(n, n * 4))
    xcsr_all = scipy.sparse.vstack(xcsrs, format="csc")
    interp2d = ycsr.dot(xcsr_all)
    return interp2d
