# Compares on fx2007 dataset LLGP SLFM for varying interpolation points
# vs COGP SLFM approx with fixed inducing points = 100

import numpy as np
from standard_tester import *

activate_logs()

if is_validation():
    import runlmc.lmc.stochastic_deriv
    runlmc.lmc.stochastic_deriv.StochasticDeriv.N_IT = 1
    runs = 1
    cogp_runs = 1
    interpolation_points = 10
    inducing_points = 10
else:
    runs = 50
    cogp_runs = 3
    interpolation_points = None
    inducing_points = 100

xss, yss, test_xss, test_yss, _, _ = foreign_exchange_2007()

np.random.seed(1234)
kgen, rgen, slfmgen, indepgen = alvarez_and_lawrence_gp()
llgp_stats = bench_runlmc(
    runs, interpolation_points, xss, yss, test_xss, test_yss, kgen, rgen,
    slfmgen, indepgen, {'verbosity': 100, 'min_grad_ratio': 0.2})
print('---> llgp Q1R2 m', interpolation_points, statsline(llgp_stats))

cogp_stats, _, _ = cogp_fx2007(cogp_runs, inducing_points)
print('---> cogp m', inducing_points, statsline(cogp_stats))

latex_table('results_fx2007.tex',
            ['LLGP', 'COGP'],
            [llgp_stats, cogp_stats])
