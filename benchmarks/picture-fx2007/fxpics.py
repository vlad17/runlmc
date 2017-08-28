import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import os
import logging
import sys
import contexttimer

import numpy as np
import scipy.linalg as la
from standard_tester import *

from runlmc.models.lmc import LMC
from runlmc.kern.rbf import RBF
from runlmc.models.optimization import AdaDelta
from runlmc.models.gpy_lmc import GPyLMC

import sys

outdir = sys.argv[1] + '/'

print('publishing results into out directory', outdir)

# Nguyen 2014 COGP uses Q=2 R=1, but that is not LMC
# Álvarez and Lawrence 2010 Convolved GP has R=4, sort of.
# Álvarez and Lawrence 2010 find that vanilla LMC works best with Q=1 R=2
# that is what we use here
ks = [RBF(name='rbf0')]
ranks = [2]
# the columns with nonzero test holdout are in test_fx
xss, yss, test_xss, test_yss, test_fx, cols = foreign_exchange_2007()

np.random.seed(1234)
lmc = LMC(xss, yss, kernels=ks, ranks=ranks)
opt = AdaDelta(verbosity=20, min_grad_ratio=0.2)
print('training LLGP')
with contexttimer.Timer() as t:
    lmc.optimize(optimizer=opt)
pred_yss, pred_vss = lmc.predict(test_xss)
llgp_time = t.elapsed
llgp_smse = smse(test_yss, pred_yss, yss)
llgp_nlpd = nlpd(test_yss, pred_yss, pred_vss)
print('    time', llgp_time, 'smse', llgp_smse, 'nlpd', llgp_nlpd)

print('training COGP')
# 1 run only, 100 inducing points, as in the paper
stats, cogp_mu, cogp_var = cogp_fx2007(1, 100)
statsnames = ['time', 'smse', 'nlpd']
print(' '.join(map(' '.join, zip(statsnames, statprintlist(stats)))))

all_xs = np.arange(min(xs.min() for xs in xss), max(xs.max()
                                                    for xs in xss) + 1)
test_ix = {col: list(cols).index(col) for col in test_fx}
pred_xss = [all_xs if col in test_fx else np.array([]) for col in cols]
lmc.prediction = 'exact'
pred_yss, pred_vss = lmc.predict(pred_xss)
pred_yss = {col: ys for col, ys in zip(cols, pred_yss)}
pred_vss = {col: vs for col, vs in zip(cols, pred_vss)}

_, axs = plt.subplots(ncols=3, figsize=(16, 4))
for col, ax in zip(test_fx, axs):

    # Prediction on entire domain for COGP
    ax.plot(all_xs, cogp_mu[col], c='black', ls='-')
    sd = np.sqrt(cogp_var[col])
    top = cogp_mu[col] + 2 * sd
    bot = cogp_mu[col] - 2 * sd
    ax.fill_between(all_xs, bot, top, facecolor='grey', alpha=0.2)

    # Prediction for LLGP
    ax.plot(all_xs, pred_yss[col], c='red')
    sd = np.sqrt(pred_vss[col])
    top = pred_yss[col] + 2 * sd
    bot = pred_yss[col] - 2 * sd
    ax.fill_between(all_xs, bot, top, facecolor='green', alpha=0.3)

    # Actual holdout
    marker_size = 5
    test_xs = test_xss[test_ix[col]]
    test_ys = test_yss[test_ix[col]]
    ax.scatter(test_xs, test_ys, c='blue',
               edgecolors='none', s=marker_size, zorder=11)

    # Rest of image (training)
    rest_xs = xss[test_ix[col]]
    rest_ys = yss[test_ix[col]]
    ax.scatter(rest_xs, rest_ys, c='magenta',
               edgecolors='none', s=marker_size, zorder=10)

    ax.set_xlim([0, 250])
    ax.set_title('output {} (95%)'.format(col))

print('fx2007graph.pdf')
plt.savefig(outdir + 'fx2007graph.pdf', format='pdf', bbox_inches='tight')
plt.clf()

print('running on-the-fly, exact, precompute predictions for comparison')
# warning: precompute takes a while, ~10 min on a 4-core 2015 laptop
methods = ['on-the-fly', 'exact', 'precompute']
results = []
resultstring = ''
for m in methods:
    lmc.prediction = m
    with contexttimer.Timer() as t:
        pred_yss, pred_vss = lmc.predict(test_xss)
    resultstring += ('method {: <12} smse {:6.4f} nlpd {:13.4e} time {}\n'
                     .format(
                         m,
                         smse(test_yss, pred_yss, yss),
                         nlpd(test_yss, pred_yss, pred_vss),
                         t.elapsed))
    results.append(np.hstack(pred_vss))


def rel_norm_diff(exact, x):
    diff = la.norm(exact - x)
    orig = la.norm(exact)
    return diff / orig


onthefly, exact, precomp = results

print('    reporting results in prediction_time.txt')
with open(outdir + 'prediction_time.txt', 'w') as f:
    f.write(resultstring)
    print('on the fly ', rel_norm_diff(exact, onthefly), file=f)
    print('precomputed', rel_norm_diff(exact, precomp), file=f)

n = len(lmc.y)
m = len(lmc.inducing_grid)
D = len(lmc.noise)
Q = len(lmc.kernels)
R = int(sum(ranks) / Q)
paramstr = 'n={},m={},D={},Q={},R={}'.format(n, m, D, Q, R)

print('running sampling prediction for different samples')
# warning: last one takes ~10 min on a 4-core 2015 laptop
samples = [100, 500, 1000]
samp_results = []
for s in samples:
    lmc.prediction = 'sample'
    lmc.variance_samples = s
    if 'sampled_nu' in lmc._cache:
        del lmc._cache['sampled_nu']
    with contexttimer.Timer() as t:
        pred_yss, pred_vss = lmc.predict(test_xss)
    with open(outdir + 'prediction_time.txt', 'a') as f:
        print('method {: <12} smse {:6.4f} nlpd {:13.4e} time {}'.format(
            'sample {:5d}'.format(s),
            smse(test_yss, pred_yss, yss),
            nlpd(test_yss, pred_yss, pred_vss),
            t.elapsed), file=f)
    samp_results.append(np.hstack(pred_vss))

plt.semilogy(exact, label='exact')
lss = [':', '--', '-']
for ns, res, ls in zip(samples, samp_results, lss):
    plt.semilogy(res, label=r'$N_s={}$'.format(ns), ls=ls)
plt.title(r'pred $\sigma^2$ for ${}$'.format(paramstr))
plt.xlabel('test point index')
plt.ylabel('predictive variance')
plt.legend(bbox_to_anchor=(1.05, 0.6), loc=2)
print('sample-pred-var.eps')
plt.savefig(outdir + 'sample-pred-var.eps', format='eps', bbox_inches='tight')
plt.clf()
