# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

# The code below was modified from MSGP demo code provided by
# Wilson. The demo code itself is based off of extending components
# in GPML, which is BSD 2-clause licenced. I've replicated the
# copyrights and/or licences to both code source in this repository's
# LICENSE file.

import numpy as np
import scipy
import scipy.sparse

def cubic_kernel(x):
    """
    The cubic convolution kernel can be used to compute interpolation
    coefficients. Its definition is taken from:

    Cubic Convolution Interpolation for Digital Image Processing by
    Robert G. Keys. It is supported on values of absolute magnitude less than
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

    :returns: the interpolation coefficient matrix
    :raises ValueError: if any of the following hold true:

        #. `grid` or `samples` are not 1-dimensional
        #. `grid` size less than 4
        #. `grid` is not equispaced
        #. `samples` is not strictly contained in `grid[1:-1]`
    """

    grid_size = len(grid)
    n_samples = samples.size

    if grid.ndim != 1:
        raise ValueError('grid dim {} should be 1'.format(grid.ndim))

    if samples.ndim != 1:
        raise ValueError('samples dim {} should be 1'.format(samples.ndim))

    if grid_size < 4:
        raise ValueError('grid size {} must be >=4'.format(grid_size))

    if samples.min() <= grid[1]:
        raise ValueError(
            'Second grid point {} must be < the min sample {}'
            .format(grid[1], samples.min()))

    if samples.max() >= grid[-2]:
        raise ValueError(
            'Penultimate grid point {} must be > the max sample {}'
            .format(grid[-2], samples.max()))

    delta = grid[1] - grid[0]
    factors = (samples - grid[0]) / delta
    # closest refers to the closest grid point that is smaller
    idx_of_closest = np.floor(factors)
    dist_to_closest = factors - idx_of_closest # in units of delta

    csr = scipy.sparse.csr_matrix((n_samples, grid_size), dtype=float)
    for conv_idx in range(-2, 2): # cubic conv window
        coeff_idx = idx_of_closest - conv_idx
        relative_dist = dist_to_closest + conv_idx
        data = cubic_kernel(relative_dist)
        col_idx = coeff_idx
        ind_ptr = np.arange(0, n_samples + 1)
        csr += scipy.sparse.csr_matrix((data, col_idx, ind_ptr),
                                       shape=(n_samples, grid_size))
    return csr
