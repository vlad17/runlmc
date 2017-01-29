# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

# pylint: skip-file

import sys

import contexttimer
import numpy as np
import scipy.linalg as la
import scipy.spatial.distance
import scipy.sparse.linalg as sla

from runlmc.approx.interpolation import multi_interpolant
from runlmc.kern.rbf import RBF
from runlmc.kern.matern32 import Matern32
from runlmc.kern.std_periodic import StdPeriodic
from runlmc.models.lmc import LMC
from runlmc.lmc.parameter_values import ParameterValues
from runlmc.lmc.grid_kernel import *
from runlmc.lmc.kernel import ExactLMCKernel, ApproxLMCKernel

_HELP_STR = """
Usage: python bench.py n_o d q eps [kern] [seed] [test-type]

n_o > 7 is the number of inputs per output
d > 0 is the number of outputs
q > 0 is the number of LMC kernel terms
eps > 0 is the constant diagonal perturbation mean (a float)
kern is the kernel type, default rbf, one of 'rbf' 'periodic' 'matern' 'mix'
seed is the random seed, default 1234
test-type performs a particular test, default inversion:
  'inversion' 'logdet' 'gradients'

For all benchmarks, this constructs a variety of LMC kernels,
all of which conform to the parameters n_o,d,q,eps specified
above. The particular kernel constructed is the sum of q ICM
terms:

  Aq = aa^T, a ~ Normal(mean=0, cov=I)
  kappa ~ vector of InverseGamma(shape=1, scale=1)
  Bq = Aq + kappa I
  Kq = one of RBF, Matern32, StdPeriodic applied to inputs
  entire term: HadamardProduct(KroneckerProduct(Bq, 1), Kq

Finally, we add independent noise for each output, sampled
from InverseGamma(shape=(1 + eps^-1), scale=1)

Choose q = d = 1 and n large to test Toeplitz, mainly
Choose q = 1 and n ~ d^2 > 7 to test Kronecker, mainly

Inputs/outputs are random and uniform in (0, 1). The interpolation grid
used by the SKI approximation is a grid with n_o datapoints.
"""

def _main():
    """Runs the benchmarking program."""
    min_args = 5
    max_args = min_args + 3
    if len(sys.argv) not in range(min_args, max_args + 1):
        print(_HELP_STR)
        sys.exit(1)

    n_o = int(sys.argv[1])
    d = int(sys.argv[2])
    q = int(sys.argv[3])
    eps = float(sys.argv[4])
    kern = sys.argv[5] if len(sys.argv) > 5 else 'rbf'
    seed = int(sys.argv[6]) if len(sys.argv) > 6 else 1234
    testtype = sys.argv[7] if len(sys.argv) > 7 else 'inversion'
    kerntypes = ['rbf', 'periodic', 'matern', 'mix']
    testtypes = ['inversion', 'logdet', 'gradients']

    assert n_o > 7
    assert d > 0
    assert q > 0
    assert eps > 0
    assert kern in kerntypes
    assert testtype in testtypes
    np.random.seed(seed)
    n = n_o * d

    print('n_o {} d {} q {} eps {} kern {} seed {} test-type {}'.format(
        n_o, d, q, eps, kern, seed, testtype))

    coreg_vecs = np.random.randn(q, d)
    coreg_diags = np.reciprocal(np.random.gamma(shape=1, scale=1, size=(q, d)))
    noise = np.reciprocal(np.random.gamma(
        shape=(1 + (1 / eps)), scale=1, size=d))
    kernels = gen_kernels(q)
    descriptions = [
        'rbf only - inverse lengthscales in logspace(0, 1, q)',
        'periodic only - inverse lengthscale is 1, periods in logspace',
        'matern32 only - invers lenngthscales in logspace',
        'mixed - all the above, with lengthscales/periods in'
        ' logspace(0, 1, max(q // 3, 1)']
    kdict = {k_name: (k, desc) for k_name, k, desc in
             zip(kerntypes, kernels, descriptions)}

    Xs, Ys = np.random.rand(2, d, n_o)

    dists, grid_dists, interpolant, interpolant_T = prep(
        d, n_o, Xs)

    k, desc = kdict[kern]
    print()
    print(desc)

    params = ParameterValues(
        coreg_vecs, coreg_diags, k, [len(X) for X in Xs], np.hstack(Ys), noise)

    run_kernel_benchmark(
        params, dists, grid_dists, interpolant, interpolant_T, testtype)

def prep(d, n_o, Xs):
    # Replicates LMC (runlmc.models.lmc) code minimally.
    with contexttimer.Timer() as exact:
        dists = scipy.spatial.distance.pdist(Xs.reshape(-1, 1))
        dists = scipy.spatial.distance.squareform(dists)
    with contexttimer.Timer() as apprx:
        grid, m = LMC._autogrid(Xs, lo=None, hi=None, m=None)
        grid_dists = grid - grid[0]
        interpolant = multi_interpolant(Xs, grid)
        interpolantT = interpolant.transpose().tocsr()

    print()
    print('preparation time (once per optimization)')
    print('    {:8.4f} sec exact - pairwise distances'
          .format(exact.elapsed))
    print('    {:8.4f} sec apprx - linear interpolation'
          .format(apprx.elapsed))

    return dists, grid_dists, interpolant, interpolantT

def run_kernel_benchmark(
        params, dists, grid_dists, interpolant, interpolantT, testtype):

    with contexttimer.Timer() as t:
        exact = ExactLMCKernel(params, dists)
    chol_time = t.elapsed
    eigs = np.fabs(np.linalg.eigvalsh(exact.K))
    with contexttimer.Timer() as t:
        apprx = ApproxLMCKernel(SumGridKernel(
            params, grid_dists, interpolant, interpolantT))
    print('    covariance matrix info')
    print('        largest  eig        {:8.4e}'.format(eigs.max()))
    print('           -> (predicted)   {:8.4e}'
          .format(apprx.K.upper_eig_bound()))
    print('        smallest eig        {:8.4e}'.format(eigs.min()))
    print('        l2 condition number {:8.4e}'
          .format(eigs.max() / eigs.min()))
    print('    matrix materialization/inversion time')
    print('        {:10.4f} sec exact - cholesky'.format(chol_time))
    print('        {:10.4f} sec apprx - solve K*alpha=y'.format(t.elapsed))

    matrix_diff = np.fabs(apprx.K.as_numpy() - exact.K).mean()
    print('        {:9.4e} |K_exact - K_apprx|_1 / n^2'.format(matrix_diff))
    alpha_diff = np.fabs(apprx.deriv.alpha - exact.deriv.alpha).mean()
    print('        {:9.4e} |alpha_exact - alpha_apprx|_1 / n'
          .format(alpha_diff))

    if testtype == 'logdet':
        print('    logdet')
        sgn, tru = np.linalg.slogdet(exact.K)
        assert sgn > 0

        # TODO: generalize logdet, abstract algorithms, use in optimization?
        # TODO: in abstracted algorithms, need more robust 'cutoffs':
        #       e.g., when convergence stalls

        def minres(mvm):
            ctr = 0
            def cb(_):
                nonlocal ctr
                ctr += 1
            m = len(apprx.grid_dists)
            n = params.n
            lo = sla.LinearOperator((n, n), matvec=mvm)

            def inv_mvm(x):
                Kinv_x, succ = sla.minres(
                    lo, x, tol=1e-10, maxiter=m, callback=cb)
                error = np.linalg.norm(x - mvm(Kinv_x))
                #print('            '
                #      'minres conv recon {:8.4e} it {:6d} m {:6d} succ {}'
                #      .format(error, ctr, m, succ))
                return Kinv_x

            return inv_mvm

        dd = np.diag(exact.K) # actual constructor harder w/o exact
        D = lambda x: dd * x
        N = lambda x: apprx.K.matvec(x) - D(x)
        delta0 = np.log(dd).sum()

        def mvm(t):
            ndtn = lambda x: D(x) + t * N(x)
            invmvm = minres(ndtn)
            return lambda x: N(invmvm(x))

        def apx_tr(mvm, n):
            rs = np.random.randint(0, 2, (n, params.n)) * 2 - 1
            trace = 0
            var = 0
            for r in rs:
                x = r.dot(mvm(r))
                trace += x
            trace /= len(rs)
            return trace

        def sample(t):
            return apx_tr(mvm(t), 2)

        vs = np.vectorize(sample)

        import scipy
        integ = scipy.integrate.fixed_quad(vs, 0, 1, n=10)[0]
        #integ, _, _, _ = scipy.integrate.quad(
        # sample, 0, 1, limit=1, full_output=1)
        trace = integ + delta0

        print('        true value from exact {:8.4e}'.format(tru))
        chol = np.log(np.diag(exact.L[0])).sum() * 2
        print('        from cholesky diag    {:8.4e}'.format(chol))
        print('        from         trace    {:8.4e}'.format(trace))
        return

    if testtype == 'inversion':
        print('    krylov subspace methods m={}'.format(len(grid_dists)))

        from runlmc.approx.iterative import Iterative
        solve = Iterative.solve

        chol = lambda y: (la.cho_solve(exact.deriv.L, y), 0)

        basic = SumGridKernel(params, grid_dists, interpolant, interpolantT)
        lcg = lambda y: solve(basic, y, verbose=True, minres=False)
        minres = lambda y: solve(basic, y, verbose=True, minres=True)

        pre = SumGridKernel(params, grid_dists, interpolant, interpolantT)
        #pre.preconditioner =
        lcgp = lambda y: solve(pre, y, verbose=True, minres=False)

        methods = [
            (chol, 'chol'),
            (lcg, 'lcg'),
            #(lcgp, 'lcg + pre'),
            (minres, 'minres')]

        for f, name in methods:
            with contexttimer.Timer() as t:
                x, it = f(params.y)
            recon_err = np.linalg.norm(params.y - exact.K.dot(x))
            print('        {:9.4e} reconstruction {:10.4f} '
                  'sec {:8d} iterations {}'
                  .format(recon_err, t.elapsed, it, name))
        return

    def check_grads(f, name):
        with contexttimer.Timer() as t:
            exact_kgrad = f(exact)
        ngrad = sum(map(len, exact_kgrad))
        print('    {} gradients # {}'.format(name, ngrad))
        print('        {:10.4f} sec exact per gradient'
              .format(t.elapsed / ngrad))
        tot_exact_time = t.elapsed
        with contexttimer.Timer() as t:
            apprx_kgrad = f(apprx)
        assert ngrad == sum(map(len, apprx_kgrad))
        print('        {:10.4f} sec apprx per gradient'
              .format(t.elapsed / ngrad))
        tot_apprx_time = t.elapsed
        exact_kgrad = np.hstack(exact_kgrad)
        apprx_kgrad = np.hstack(exact_kgrad)
        err = exact_kgrad - apprx_kgrad
        print('        {:9.4e} avg grad error'.format(np.fabs(err).mean()))
        print('        {:9.4e} avg signed error'.format(err.mean()))
        return err, tot_exact_time, tot_apprx_time

    gradient_type = [
        (lambda x: x.kernel_gradients(), 'kernel'),
        (lambda x: x.coreg_vec_gradients(), 'coregionalization Aq'),
        (lambda x: x.coreg_diag_gradients(), 'coregionalization kappa'),
        (lambda x: [x.noise_gradient()], 'noise')]

    errs = np.array([])
    tot_exact_time = 0
    tot_apprx_time = 0
    for f, name in gradient_type:
        err, exact_time, apprx_time = check_grads(f, name)
        errs = np.append(errs, err)
        tot_exact_time += exact_time
        tot_apprx_time += apprx_time

    print('    total gradient runtime summary')
    print('        {:10.4f} sec exact all gradients'.format(tot_exact_time))
    print('        {:10.4f} sec apprx all gradients'.format(tot_apprx_time))
    print('        {:9.4e} avg grad error'.format(np.fabs(errs).mean()))
    print('        {:9.4e} avg signed error'.format(errs.mean()))

def gen_kernels(q):
    kern_funcs = [RBF, lambda period: StdPeriodic(1, period), Matern32]
    kernels = [[kfunc(gamma)
                for gamma in np.logspace(0, 1, q)]
               for kfunc in kern_funcs]
    kernels.append([kfunc(gamma)
                    for gamma in np.logspace(0, 1, max(q // 3, 1))
                    for kfunc in kern_funcs])
    return kernels

if __name__ == '__main__':
    _main()
