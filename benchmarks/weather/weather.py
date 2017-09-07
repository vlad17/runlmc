# compares on 15K weather dataset LLGP SLFM reduction vs COGP SLFM approx

import os
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
from standard_tester import *

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


class Suite:
    def __init__(self, num_interp=750):
        self.num_interp = 750

    def setup_cache(self):
        xss, yss, test_xss, test_yss, _ = weather()

        np.random.seed(1234)
        # rank 2 SLFM, same as Q=2 model of COGP
        kgen, rgen, slfmgen, indepgen = slfm_gp(len(xss), 2)
        llgp_stats = bench_runlmc(
            runs, self.num_interp, xss, yss, test_xss, test_yss, kgen, rgen,
            slfmgen, indepgen, {'verbosity': 100})
        return llgp_stats

    def track_mean_time(self, llgp_stats):
        (mean_time, _), _, _ = llgp_stats
        return mean_time

    def track_se_time(self, llgp_stats):
        (_, se_time), _, _ = llgp_stats
        return se_time

    def track_mean_smse(self, llgp_stats):
        _, (mean_smse, _), _ = llgp_stats
        return mean_smse

    def track_se_smse(self, llgp_stats):
        _, (_, se_smse), _ = llgp_stats
        return se_smse

    def track_mean_nlpd(self, llgp_stats):
        _, _, (mean_nlpd, _) = llgp_stats
        return mean_nlpd

    def track_se_nlpd(self, llgp_stats):
        _, _, (_, se_nlpd) = llgp_stats
        return se_nlpd


Suite.setup_cache.timeout = 3600 * 12
Suite.track_mean_time.benchmark_name = 'weather time mean'
Suite.track_mean_time.units = 'seconds'
Suite.track_se_time.benchmark_name = 'weather time standard error'
Suite.track_se_time.units = 'seconds'
Suite.track_mean_smse.benchmark_name = 'weather smse mean'
Suite.track_se_smse.benchmark_name = 'weather smse standard error'
Suite.track_mean_nlpd.benchmark_name = 'weather nlpd mean'
Suite.track_se_nlpd.benchmark_name = 'weather nlpd standard error'


def make_llgp_colname(interp):
    return r'\begin{tabular}{c}LLGP\\$m=' + str(interp) + r'$\end{tabular}'


def main():
    llgp_stats = []
    for num_interp in interpolating_points:
        stats = Suite(num_interp).setup_cache()
        llgp_stats.append(stats)
        print('---> llgp slfm m', num_interp, statsline(stats))

    cogp_stats, _, _ = cogp_weather(cogp_runs, inducing_points)
    print('---> cogp m', inducing_points, statsline(cogp_stats))

    colnames = [make_llgp_colname(interpolating_points[0]),
                make_llgp_colname(interpolating_points[-1]),
                'COGP']
    latex_table('results_weather.tex',
                colnames,
                [llgp_stats[0], llgp_stats[-1], cogp_stats])
