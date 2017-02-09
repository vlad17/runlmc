import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


from standard_tester import *

from runlmc.models.lmc import LMC
from runlmc.kern.rbf import RBF
from runlmc.kern.matern32 import Matern32
from runlmc.kern.std_periodic import StdPeriodic
from runlmc.models.optimization import AdaDelta
from runlmc.models.gpy_lmc import GPyLMC

np.random.seed(1234)
xss, yss, test_xss, test_yss, cols = foreign_exchange_33k()
import logging
from runlmc.models.lmc import LMC, _LOG
logging.getLogger().addHandler(logging.StreamHandler())
_LOG.setLevel(logging.INFO)

nk = 4
m = 300
ks = [RBF(name='rbf{}'.format(i)) for i in range(nk)]
ranks= [1 for _ in range(nk)]
lmc = LMC(xss, yss, kernels=ks, ranks=ranks,
          normalize=True, m=m, max_procs=30)


def cb():
    global lmc, test_xss, test_yss, pred_vss
    with contexttimer.Timer() as t:
        pred_yss, pred_vss = lmc.predict(test_xss)
    print('    pred time', t.elapsed, 'smse', smse(test_yss, pred_yss, yss), 'nlpd', nlpd(test_yss, pred_yss, pred_vss))
    print('    noise', ' '.join(['{:8.2f}'.format(e) for e in lmc.noise]))
    np.save('f33ksave.npy', lmc.param_array)


opt = AdaDelta(verbosity=100, callback=cb)

with contexttimer.Timer() as t:
    lmc.optimize(optimizer=opt)
print('opt time', t.elapsed)

np.save('f33ksave.npy', lmc.param_array)
