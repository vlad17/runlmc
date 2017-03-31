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
    inducing_points = 10
else:
    runs = 10
    cogp_runs = 10
    interpolating_points = [500, 600, 700, 800, 900, 1000]
    inducing_points = 200

llgp_stats = []
for num_interp in interpolating_points:
    np.random.seed(1234)
    # rank 2 SLFM, same as Q=2 model of COGP
    kgen, rgen, slfmgen, indepgen = slfm_gp(len(xss), 2)
    stats = bench_runlmc(
        runs, num_interp, xss, yss, test_xss, test_yss, kgen, rgen,
        slfmgen, indepgen, {'verbosity': 100})
    print('---> llgp slfm m', num_interp, statsline(stats))
    llgp_stats.append(stats)

cogp_stats, _, _ = cogp_weather(cogp_runs, inducing_points)
print('---> cogp m', inducing_points, statsline(cogp_stats))

def make_llgp_colname(interp):
    return r'\begin{tabular}{c}LLGP\\$m=' + str(interp) + r'$\end{tabular}'

colnames = [make_llgp_colname(interpolating_points[0]),
            make_llgp_colname(interpolating_points[-1]),
            'COGP']
latex_table('results_weather.tex',
            colnames,
            [llgp_stats[0], llgp_stats[-1], cogp_stats])
