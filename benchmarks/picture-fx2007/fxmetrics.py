import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import os
import logging
import sys
import contexttimer

import numpy as np
from standard_tester import *

from runlmc.models.lmc import LMC
from runlmc.kern.rbf import RBF
from runlmc.models.optimization import AdaDelta
from runlmc.models.gpy_lmc import GPyLMC

import sys

outdir = sys.argv[1] + '/'

ks = [RBF(name='rbf0')]
ranks = [2]
xss, yss, test_xss, test_yss, test_fx, cols = foreign_exchange_2007()

print('running MINRES metrics')

# also takes ~10 min
np.random.seed(1234)
lmc_with_metrics = LMC(xss, yss, kernels=ks, ranks=ranks, metrics=True, max_procs=1)
lmc_with_metrics.optimize(optimizer=AdaDelta(
        # Force full 35 iterations with ratio = 0
        verbosity=10, max_it=35, min_grad_ratio=0))

def moving_average(a, n) :
    sums = np.add.accumulate(a, dtype=float)
    sums[n:] = sums[n:] - sums[:-n]
    sums[n:] /= n
    sums[:n] /= np.arange(1, n + 1)
    return sums

def div_rolled_max(a, n):
    ma = moving_average(a, n)
    roll_max = np.maximum.accumulate(ma)
    return roll_max

plt.plot(lmc_with_metrics.metrics.iterations, label='minres MVMs')
n = len(lmc_with_metrics.y)
plt.axhline(n, label='iterartion cutoff ($n={}$)'.format(n), c='r')
plt.xlabel('optimization iteration')
plt.ylabel('inversion iterations')
plt.legend(bbox_to_anchor=(.3, -0.15), loc=2)
print('iterations.eps')
plt.savefig(outdir + 'iterations.eps', format='eps', bbox_inches='tight')
plt.clf()

fig, ax1 = plt.subplots()
ax1.plot(lmc_with_metrics.metrics.log_likely, c='r')
ax1.set_ylabel('log likelihood', color='r')
ax2 = ax1.twinx()
ax2.plot(lmc_with_metrics.metrics.grad_norms, c='b', label='raw norms')

# Roll = 1 by default
ax2.plot(0.2 * div_rolled_max(lmc_with_metrics.metrics.grad_norms, 1),
         c='g', ls='--',
         label='min grad cutoff')
ax2.set_ylabel('grad norms', color='b')
fig.tight_layout()
plt.legend(bbox_to_anchor=(.3, -.07), loc=2)
print('running_cutoff.eps')
plt.savefig(outdir + 'running_cutoff.eps', format='eps', bbox_inches='tight')
plt.clf()
