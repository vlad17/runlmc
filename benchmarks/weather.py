# Run this as follows, in benchmarks/
# OMP_NUM_THREADS=1 PYTHONPATH=.:.. python -u weather.py 2>&1 | tee example-stdout-weather.txt | egrep -e '--->|launched'
import os
import logging
import sys

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

np.random.seed(1234)

xss, yss, test_xss, test_yss, cols = weather()

runs = 10

import os

with Pool(cpu_count()) as pool:
    workers = pool.starmap(os.getpid, [[] for _ in range(4 * cpu_count())])
    workers = set(workers)
    print(len(workers), 'distinct workers launched')

    for num_interp in [1000, 2000, 3000, 4000, 5000]:
        kgen = lambda: []
        rgen = lambda: []
        slfmgen = lambda: [RBF(name='slfm0'), RBF(name='slfm1')]
        indepgen = lambda: [Scaled(RBF()) for _ in xss]
        llgp_time, llgp_smse, llgp_nlpd, lmc = runlmc(
            runs, num_interp, xss, yss, test_xss, test_yss, kgen, rgen,
            slfmgen, indepgen, {'verbosity': 100}, extrapool=pool)
        print('---> llgp slfm m', num_interp, 'time', statprint(llgp_time), 'smse', statprint(llgp_smse), 'nlpd', statprint(llgp_nlpd))

for num_induc in [100, 200, 300]:
    cogp_time, cogp_smse, cogp_nlpd, _, _ = cogp_weather(runs, num_induc)
    print('---> cogp m', num_induc, 'time', cogp_time, 'smse', cogp_smse, 'nlpd', cogp_nlpd)
