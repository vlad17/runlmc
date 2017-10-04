# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import numpy as np
import scipy.stats
from paramz.transformations import Logexp

from ..parameterization.param import Param
from ..parameterization.parameterized import Parameterized


class FunctionalKernel(Parameterized):
    """
    An LMC kernel can be specified by the number of latent GP kernels
    it contains. Recall a full LMC kernel defines the similarity between
    two inputs :math:`\\textbf{x}_i,\\textbf{x}_j` belonging to two outputs
    :math:`a,b`, respectively, as follows (noise not included)

    .. math::

        K((\\textbf{x}_i, a),(\\textbf{x}_j, b)) = \sum_{q=1}^Q B_{ab}^{(q)}
            k_q(\\textbf{x}_i,\\textbf{x}_j)

    If we enumerate all inputs across all our :math:`D` outputs
    :math:`\\{z_j\\}_j=\\{( \\textbf{x}_i, a)|a\\in [D]\\}`, then the complete
    LMC kernel evaluated as single matrix over an entire multi-output dataset
    :math:`X` gives :math:`K_{X,X}\\in\\mathbb{R}^{n\\times n}`, with $mn$-th
    entry :math:`(K_{X,X})_{mn}=K(z_m,z_n)`.

    Since we can perform certain optimizations if :math:`B^{(q)}` contains a
    single nonzero diagonal entry or is of single rank. We refer to this as
    the coregionalization matrix for stationary subkernel :math:`k_q`.

    `FunctionalKernel` provides a convenient wrapper for specifying such
    kernels, which should all be instances of
    :class:`runlmc.kern.stationary_kernel.StationaryKern`. This class is not
    tied to any data :math:`X` but represents the :math:`K` function, not
    matrix, above. This class is, however, tied to parameters.
    Especially important is the dichotomy with
    :class:`runlmc.lmc.likelihood.LMCLikelihood`, which is a fixed evaluation
    of a `FunctionalKernel` with a fixed set of parameters on fixed data.

    After a successful initialization, we have
    `Q == len(lmc_kernels) + len(slfm_kernels) + len(indep_gp)` and
    `len(indep_gp) == D`. Each :math:`A_q,\\boldsymbol\\kappa_q`, becomes
    this model, with name `a<q>`, where `<q>` is replaced with a specific
    number.

    Before use, input dimension should be specified with :meth:`set_input_dim`.
    This is usually done automatically by the model, such as
    :class:`runlmc.models.interpolated_llgp.InterpolatedLLGP`.

    :param D: number of outputs
    :param lmc_kernels: a list of kernels for which the corresponding
        coregionalization matrix has full rank
    :param lmc_ranks: a list of integers of the same length as `lmc_kernels`
        each with value :math:`r_q` at least 1 which specify that
        the coregionalization matrix for the corresponding kernel
        :math:`k_q` in the `lmc_kernels` list can be decomposed as
        :math:`B^{(q)}=A_qA_q^{ \\top } + \\mathop{\\text{diag}}
        \\boldsymbol\\kappa_q`, with :math:`A_q` of rank :math:`r_q`.
    :param slfm_kernels: an SLFM kernel restricts its coregionalization
        matrix to a single rank :math:`A_qA_q^\\top`
    :param indep_gp: indpedent GPs for each output :math:`i`,
        with associated coregionalization matrices
        :math:`\\textbf{e}_i\\textbf{e}_i^\\top`.
    :param indep_gp_index: should be the same length as `indep_gp`, and
        specifies which output the kernel in the `indep_gp` list in
        the same place as an index is associated with. Defaults to
        `range(len(indep_gp))`.
    :param name: :mod:`paramz` name for this kernel
    :raises ValueError: if any of the parameters don't meet the above
        requirements, or `D,Q` are unspecified, 0, or inconsistent.
    :ivar Q: `Q`, subkernel count including SLFM and indpendent kernels
    :ivar D: `D`, output dimension
    :ivar num_lmc: number of LMC kernels (a dictionary, where the key
       is the active dimensions and the value is the number of LMC kernels
       for that set of active dimensions)
    :ivar num_slfm: number of SLFM kernels, as `num_lmc`
    :ivar num_indep: number of independent GP kernels, as `num_lmc`
    :ivar active_dims: a dictionary whose keys are the subsets of the full
        input dimension set `{1, ..., P}`. Only defined after
        :meth:`set_input_dim` has been called.
    """

    def __init__(self, D=None, lmc_kernels=None, lmc_ranks=None,
                 slfm_kernels=None, indep_gp=None, indep_gp_index=None,
                 name='kern'):
        super().__init__(name=name)

        if not D:
            raise ValueError('D should be specified')
        self.D = D

        if not lmc_kernels and not slfm_kernels and not indep_gp:
            raise ValueError('Number of kernels should be >0')

        if len(lmc_kernels) != len(lmc_ranks):
            raise ValueError('# LMC kernels should equal # LMC ranks')

        if not all(map(lambda rank: rank > 0, lmc_ranks)):
            raise ValueError('LMC ranks not positive')

        lmc_kernels = lmc_kernels or []
        slfm_kernels = slfm_kernels or []
        indep_gp = indep_gp or []
        indep_gp_index = indep_gp_index or range(len(indep_gp))

        if len(indep_gp) != len(indep_gp_index):
            raise ValueError('indep GP number of kernels should match indices')

        self._kernels = lmc_kernels + slfm_kernels + indep_gp
        for k in self._kernels:
            self.link_parameter(k)

        self._coreg_vecs = self._initialize_Aq(
            lmc_ranks, len(slfm_kernels), len(indep_gp))
        self._coreg_diags = self._initialize_kq(
            len(lmc_kernels), len(slfm_kernels), indep_gp_index)
        self.P = None
        self.active_dims = {}
        self.num_lmc = {}
        self.num_slfm = {}
        self.num_indep = {}

        self._num_lmc = len(lmc_kernels)
        self._num_slfm = len(slfm_kernels)

        # Corresponds to epsilon
        self._noise = Param('noise', 0.1 * np.ones(self.D), Logexp())
        self.link_parameter(self._noise)

    _TRUNCNORM = scipy.stats.truncnorm(-1, 1)

    @staticmethod
    def _randinit(sx, sy):
        return FunctionalKernel._TRUNCNORM.rvs(size=(sx, sy))

    @staticmethod
    def _count(dictionary, key):
        dictionary.setdefault(key, 0)
        dictionary[key] += 1

    def set_input_dim(self, P):
        """Set the input dimension for the kernel."""
        if self.P == P:
            return
        if self.P is not None:
            raise ValueError('Cannot set input dimension twice')
        self.P = P
        all_dims = tuple(range(P))
        for i, k in enumerate(self._kernels):
            if k.active_dims is None:
                k.active_dims = all_dims
            else:
                k.active_dims = tuple(sorted(list(k.active_dims)))
            self.active_dims.setdefault(k.active_dims, []).append(i)
            if i < self._num_lmc:
                FunctionalKernel._count(self.num_lmc, k.active_dims)
            elif i < self._num_lmc + self._num_slfm:
                FunctionalKernel._count(self.num_slfm, k.active_dims)
            else:
                FunctionalKernel._count(self.num_indep, k.active_dims)
        for d in (self.num_lmc, self.num_slfm, self.num_indep):
            for ad in self.active_dims:
                if ad not in d:
                    d[ad] = 0

    def _initialize_Aq(self, lmc_ranks, num_slfm, num_indep):
        """Initialize, link, and return coregionalization vectors A_q for all
        Q kernels, for lmc, slfm, and inependent kernels, in that order."""

        coreg_vecs = []
        initial_vecs = []
        initial_vecs += [FunctionalKernel._randinit(rank, self.D)
                         for rank in lmc_ranks]
        initial_vecs += [FunctionalKernel._randinit(1, self.D)
                         for _ in range(num_slfm)]
        initial_vecs += [np.zeros((1, self.D))
                         for _ in range(num_indep)]
        for i, coreg_vec in enumerate(initial_vecs):
            coreg_vecs.append(Param('a{}'.format(i), coreg_vec))
            # independent kernels have no off-diagonal coregionalization
            # and it certainly isn't modifiable during optimization
            if i < len(lmc_ranks) + num_slfm:
                self.link_parameter(coreg_vecs[-1])

        return coreg_vecs

    def _initialize_kq(self, num_lmc, num_slfm, indep_gp_index):
        """Initializes kappa_q analogously as Aq in _initialize_Aq()"""
        coreg_diags = []
        for _ in range(num_lmc):
            i = len(coreg_diags)
            coreg_diag = np.ones(self.D)
            coreg_diags.append(
                Param('kappa{}'.format(i), coreg_diag, Logexp()))
            self.link_parameter(coreg_diags[-1])
        for _ in range(num_slfm):
            i = len(coreg_diags)
            coreg_diag = np.zeros(self.D)
            coreg_diags.append(Param('kappa{}'.format(i), coreg_diag))
            coreg_diags[-1].constrain_fixed()
        for d in indep_gp_index:
            i = len(coreg_diags)
            coreg_diag = np.zeros(self.D)
            coreg_diag[d] = 1
            coreg_diags.append(Param('kappa{}'.format(i), coreg_diag))
            coreg_diags[-1].constrain_fixed()
        return coreg_diags

    def update_gradient(self, grads):
        """Update the gradients of parameters in the functional kernel
        with respect to those calculated by a concrete `LMCLikelihood` given
        data."""
        assert self.P
        for x, dx in zip(self._coreg_vecs, grads.coreg_vec_gradients()):
            x.gradient = dx
        for x, dx in zip(self._coreg_diags, grads.coreg_diags_gradients()):
            x.gradient = dx
        for k, dk in zip(self._kernels, grads.kernel_gradients()):
            k.update_gradient(dk)
        self._noise.gradient = grads.noise_gradient()

    def total_rank(self, active_dim):
        """Total (added) coregionalization rank for all B_q matrices"""
        assert self.P
        rank = 0
        for kidx in self.active_dims[active_dim]:
            if kidx < self._num_lmc + self._num_slfm:
                rank += len(self._coreg_vecs[kidx])
        return rank

    def eval_kernels(self, dists):
        """Computes the array of k_q applied to each distance in `dists`, where
        `dists` should be a dict of `active_dim`-keyed distances."""
        assert self.P
        return np.array([k.from_dist(dists[k.active_dims])
                         for k in self._kernels])

    def eval_kernels_fixed_dim(self, dists, active_dim):
        """Computes the array of k_q applied to each distance in `dists`,
        where only kernels with the passed-in active dimensions are
        evaluated."""
        return np.array([self._kernels[kidx].from_dist(dists)
                         for kidx in self.active_dims[active_dim]])

    def eval_kernel_gradients(self, dists):
        """Computes the list of grad k_q applied to each distance in `dists`,
        where `dists` should be a dict of `active_dim`-keyed distances."""
        assert self.P
        return [k.kernel_gradient(dists[k.active_dims]) for k in self._kernels]

    @property
    def noise(self):
        return self._noise.values

    @noise.setter
    def noise(self, value):
        self._noise[:] = value

    @property
    def coreg_vecs(self):
        return [coreg_vec.values for coreg_vec in self._coreg_vecs]

    @coreg_vecs.setter
    def coreg_vecs(self, values):
        for coreg_vec, value in zip(self._coreg_vecs, values):
            coreg_vec[:] = value

    @property
    def coreg_diags(self):
        return [coreg_diag.values for coreg_diag in self._coreg_diags]

    @coreg_diags.setter
    def coreg_diags(self, values):
        for coreg_diag, value in zip(self._coreg_diags, values):
            coreg_diag[:] = value

    def coreg_mats(self, active_dim=None):
        cv = self.coreg_vecs
        cd = self.coreg_diags
        if active_dim is not None:
            idxs = self.active_dims[active_dim]
            cv = [cv[idx] for idx in idxs]
            cd = [cd[idx] for idx in idxs]
        return [a.T.dot(a) + np.diag(k) for a, k in zip(cv, cd)]

    @property
    def Q(self):
        return len(self._kernels)

    def get_active_dims(self, q):
        return self._kernels[q].active_dims
