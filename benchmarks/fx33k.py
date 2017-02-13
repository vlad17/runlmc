import os
import logging
import sys

import numpy as np
from standard_tester import *

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
xss, yss, test_xss, test_yss, cols = foreign_exchange_33k()

m=None
nk = 4
ks = [RBF(name='rbf{}'.format(i)) for i in range(nk)]
ranks= [1 for _ in range(nk)]
lmc = LMC(xss, yss, kernels=ks, ranks=ranks,
          normalize=True, m=m, max_procs=20)
prev_time = None
it = 0

def cb():
    global lmc, test_xss, test_yss, t, prev_time, it, yss

    new_time = t.elapsed
    print('    noise avg', np.mean(lmc.noise))
    print('    iteration sec', new_time - prev_time)
    prev_time = new_time
    it += 1
    np.save('/data/vyf/f33k-m-None-q-4/it{}'.format(it), lmc.param_array)

opt = AdaDelta(verbosity=100, callback=cb)

with contexttimer.Timer() as t:
    prev_time = t.elapsed
    lmc.optimize(optimizer=opt)
print('opt time', t.elapsed)

np.save('/data/vyf/f33k-m-None-q-4/final', lmc.param_array)

lmc.prediction='precompute'
