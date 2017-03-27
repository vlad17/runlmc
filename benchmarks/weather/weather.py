# compares on 15K weather dataset LLGP SLFM reduction vs COGP SLFM approx

import sys
is_validation = sys.argv[1] == 'true'


if is_validation:
    import runlmc.lmc.stochastic_deriv
    runlmc.lmc.stochastic_deriv.StochasticDeriv.N_IT = 1
    runs = 1
    cogp_runs = 1
    interpolating_points = [10]
    inducing_points = [10]
else:
    runs = 50
    cogp_runs = 3
    interpolating_points = [500, 600, 700, 800, 900, 1000]
    inducing_points = [200] # COGP

import os
import logging

import numpy as np
from standard_tester import *

from multiprocessing import Pool, cpu_count

from runlmc.models.lmc import LMC
from runlmc.kern.rbf import RBF
from runlmc.kern.scaled import Scaled
from runlmc.models.optimization import AdaDelta
from runlmc.models.gpy_lmc import GPyLMC

from runlmc.models.lmc import _LOG
from runlmc.approx.iterative import _LOG as _LOG2
logging.getLogger().addHandler(logging.StreamHandler())
_LOG.setLevel(logging.INFO)
_LOG2.setLevel(logging.INFO)

xss, yss, test_xss, test_yss, cols = weather()

import os

for num_interp in interpolating_points:
    kgen = lambda: []
    rgen = lambda: []
    slfmgen = lambda: [RBF(name='slfm0'), RBF(name='slfm1')]
    indepgen = lambda: [Scaled(RBF()) for _ in xss]
    np.random.seed(1234)
    llgp_time, llgp_smse, llgp_nlpd, lmc = runlmc(
        runs, num_interp, xss, yss, test_xss, test_yss, kgen, rgen,
        slfmgen, indepgen, {'verbosity': 100})
    print('---> llgp slfm m', len(lmc.inducing_grid), 'time', statprint(llgp_time), 'smse', statprint(llgp_smse), 'nlpd', statprint(llgp_nlpd))

for num_induc in inducing_points:
    stats, _, _ = cogp_weather(cogp_runs, num_induc)
    cogp_time, cogp_smse, cogp_nlpd = statprintlist(stats)
    print('---> cogp m', num_induc,
          'time', cogp_time, 'smse', cogp_smse, 'nlpd', cogp_nlpd)
