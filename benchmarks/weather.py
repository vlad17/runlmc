import os
import logging
import sys

import numpy as np
from standard_tester import *

from multiprocessing import Pool, cpu_count

from runlmc.models.lmc import LMC
from runlmc.kern.rbf import RBF
from runlmc.models.optimization import AdaDelta
from runlmc.models.gpy_lmc import GPyLMC

from runlmc.models.lmc import _LOG
from runlmc.approx.iterative import _LOG as _LOG2
logging.getLogger().addHandler(logging.StreamHandler())
_LOG.setLevel(logging.INFO)
_LOG2.setLevel(logging.INFO)

np.random.seed(1234)

xss, yss, test_xss, test_yss, cols = weather()

# TODO - more runs
runs = 1

num_interp = 4000

import os

with Pool(cpu_count()) as pool:
    workers = pool.starmap(os.getpid, [[] for _ in range(4 * cpu_count())])
    workers = set(workers)
    print(len(workers), 'distinct workers launched')

    ks = [RBF(name='rbf0')]
    ranks = [2]
    llgp_time, llgp_smse, llgp_nlpd, lmc = runlmc(
        runs, num_interp, xss, yss, test_xss, test_yss,
        ks, ranks, {'verbosity': 1}, extrapool=pool)

    print('llgp Q1R2 m', num_interp, 'time', llgp_time, 'smse', llgp_smse, 'nlpd', llgp_nlpd)

    ks = [RBF(name='rbf0')]
    ranks = [3]
    llgp_time, llgp_smse, llgp_nlpd, lmc = runlmc(
        runs, num_interp, xss, yss, test_xss, test_yss,
        ks, ranks, {'verbosity': 1}, extrapool=pool)

    print('llgp Q1R3 m', num_interp, 'time', llgp_time, 'smse', llgp_smse, 'nlpd', llgp_nlpd)

    ks = [RBF(name='rbf0')]
    ranks = [4]
    llgp_time, llgp_smse, llgp_nlpd, lmc = runlmc(
        runs, num_interp, xss, yss, test_xss, test_yss,
        ks, ranks, {'verbosity': 1}, extrapool=pool)

    print('llgp Q1R4 m', num_interp, 'time', llgp_time, 'smse', llgp_smse, 'nlpd', llgp_nlpd)

num_induc = 100
cogp_time, cogp_smse, cogp_nlpd, _, _ = cogp_weather(runs, num_induc)
print('cogp m', num_induc, 'time', cogp_time, 'smse', cogp_smse, 'nlpd', cogp_nlpd)
