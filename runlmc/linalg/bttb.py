# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import logging

import numpy as np
import scipy.linalg as la

from .matrix import Matrix
from ..util.docs import inherit_doc
from ..util.numpy_convenience import chunks

_LOG = logging.getLogger(__name__)


@inherit_doc
class BTTB(Matrix):
    """
    Creates a class with a parsimonious representation of a symmetric,
    square block-Toeplitz matrix of Toeplitz blocks (BTTB).

    The Toeplitz blocks requirement implies each block of this matrix is
    a Toeplitz matrix. See :class:`runlmc.linalg.toeplitz.Toeplitz` for
    a description of how the top row of a Toeplitz matrix specifies the rest
    of it.

    The block requirement specifies that each sub-block is further replicated
    in a Toeplitz manner. These matrices typically form when we have
    a stationary kernel applied to a multidimentional grid.

    For instance, suppose we have a two dimensional grid of size
    :math:`x\\times y`.

    For a fixed :math:`x_i,x_j`, we have a one-dimensional square Toeplitz
    matrix of order :math:`y` for pairs :math:`(x_i, y_k),(x_j,y_m)` and
    variable :math:`k,m\in[y]`. It is easy to see how this matrix is identical
    for :math:`x_{i+1},x_{j+1}` if the :math:`x` values are on a grid.

    For higher dimensions, e.g., three, we can correspondingly construct
    block-Toeplitz matrices of BTTB blocks. Let's call these third order BTTB
    matrices (with first order BTTB being Toeplitz). This class generalizes
    all BTTB orders.

    Fix a :math:`P`-dimensional grid of points :math:`U` with points
    :math:`\\{\\textbf{z}_{i_1i_2\cdots i_P}\\}` where the multidimensional
    index :math:`i_1i_2\cdots i_P` is flattened with :math:`P` the
    fastest-changing dimension. Since we are on a grid, any point has an
    explicit form for fixed, positive :math:`\\Delta_i` and standard basis
    vectors :math:`\\textbf{e}_i`:

    .. math::

        \\textbf{z}_{i_1i_2\cdots i_P}=\\textbf{z}_{\\textbf{0}}+\\sum_{p=1}^P
        \\Delta_pi_p\\textbf{e}_p

    Assuming we have grid size :math:`n_p` along the :math:`p`-th dimension,
    with :math:`N=\\prod_pn_p` the total grid size, we thus assume to have
    a stationary kernel :math:`k` evaluated at distances
    :math:`k(\\textbf{z}_j'-\\textbf{z}_0')` which make up element :math:`j`
    of the parameter `top`, with the straightforward index-flattening
    conversion:

    .. math::

        \\textbf{z}_{j}'=\\textbf{z}_{i_1i_2\\cdots i_P}
        \\big|_{i_p=j\\bmod n_p'};
        \;\;\;\;\;n_p'=\prod_{p'=p}^Pn_{p'}

    We can still meaningfully define the BTTB without the context of the
    kernel, with `top` the flattened version of the nested `len(sizes)`-order
    BTTB matrix first row. In other words, we have the symmetric Toeplitz
    extension of :math:`N/n_P` Toeplitz matrices.

    For details, see Fast multiplication of a recursive block Toeplitz matrix
    by a vector and its application by David Lee 1986.

    :param top: 1-dimensional :mod:`numpy` array, used as the underlying
                storage, which represents the first row :math:`t_{1j}`.
                Should be castable to a float64.
    :param sizes: array of :math:`n_p`.
    :raises ValueError: if `top` or `shape` aren't of the right shape or
         are empty.
    """

    def __init__(self, top, sizes):
        sizes = np.asarray(sizes)
        if top.shape != (len(top),):
            raise ValueError('top shape {} is not 1D'.format(top.shape))
        if not top.size:
            raise ValueError('top is empty')
        if sizes.shape != (len(sizes),):
            raise ValueError('sizes shape {} is not 1D'.format(sizes.shape))
        if np.prod(sizes) != top.size:
            raise ValueError("sizes {} don't match grid size {}"
                             .format(sizes, top.size))
        super().__init__(len(top), len(top))

        self.top = top.astype('float64', casting='safe')
        self._sizes = sizes
        self._subrectangle = tuple([slice(0, n) for n in self._sizes])
        circ = BTTB._cyclic_extend_n(top, sizes, self._subrectangle)
        self._circ_fft = np.fft.rfftn(circ)

    @staticmethod
    def _cyclic_extend_n(x, sizes, mask):
        extended = np.zeros(sizes * 2)
        extended[mask] = x.reshape(sizes)
        slices = tuple()
        for n in reversed(sizes):
            dst = slice(n + 1, 2 * n)
            src = slice(n - 1, 0, -1)
            extended[(..., dst, *slices)] = extended[(..., src, *slices)]
            slices = (slice(None), *slices)
        return extended

    @staticmethod
    def _toep_replicate(blocks, n):
        if n == 1:
            return la.toeplitz(blocks)
        nblocks = len(blocks)
        z = np.zeros((nblocks * n, nblocks * n))
        for i, tblock in enumerate(blocks):
            for j in range(nblocks - i):
                blockx, blocky = (i + j) * n, j * n
                z[blockx:blockx + n, blocky:blocky + n] = tblock
                z[blocky:blocky + n, blockx:blockx + n] = tblock.T
        return z

    def as_numpy(self):
        blocks = self.top
        blocksize = 1
        for n in reversed(self._sizes):
            blocks = chunks(blocks, n)
            blocks = [BTTB._toep_replicate(b, blocksize) for b in blocks]
            blocksize *= n
        return blocks[0]

    def matvec(self, x):
        x_fft = np.fft.rfftn(x.reshape(self._sizes), s=(self._sizes * 2))
        x_fft *= self._circ_fft
        x_ifft = np.fft.irfftn(x_fft)
        return x_ifft[self._subrectangle].ravel()

    def __str__(self):
        if len(self.top) > 50:
            topstr = 'shape {}'.format(len(self._sizes))
        else:
            topstr = '\n' + str(self.top.reshape(self._sizes))
        return 'BTTB on grid ' + topstr
