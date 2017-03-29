# compares on 15K weather dataset LLGP SLFM reduction vs COGP SLFM approx

import numpy as np
from standard_tester import *

xss, yss, test_xss, test_yss, _ = weather()

activate_logs()

if is_validation():
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
    inducing_points = [200]

for num_interp in interpolating_points:
    np.random.seed(1234)
    # rank 2 SLFM, same as Q=2 model of COGP
    kgen, rgen, slfmgen, indepgen = slfm_gp(len(xss), 2)
    llgp_time, llgp_smse, llgp_nlpd, lmc = bench_runlmc(
        runs, num_interp, xss, yss, test_xss, test_yss, kgen, rgen,
        slfmgen, indepgen, {'verbosity': 100})
    print('---> llgp slfm m', len(lmc.inducing_grid), 'time', statprint(llgp_time), 'smse', statprint(llgp_smse), 'nlpd', statprint(llgp_nlpd))

for num_induc in inducing_points:
    stats, _, _ = cogp_weather(cogp_runs, num_induc)
    cogp_time, cogp_smse, cogp_nlpd = statprintlist(stats)
    print('---> cogp m', num_induc,
          'time', cogp_time, 'smse', cogp_smse, 'nlpd', cogp_nlpd)
