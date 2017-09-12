# Compares on fx2007 dataset LLGP SLFM for varying interpolation points
# vs COGP SLFM approx with fixed inducing points = 100

import os
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
from standard_tester import *

class Suite:
    def __init__(self, runs=10, interpolation_points=None, nthreads=4):
        self.runs = runs
        self.interpolation_points = interpolation_points
        self.nthreads = nthreads if nthreads else None

    def setup_cache(self):
        xss, yss, test_xss, test_yss, _, _ = foreign_exchange_2007()

        np.random.seed(1234)
        kgen, rgen, slfmgen, indepgen = alvarez_and_lawrence_gp()
        llgp_stats = bench_runlmc(
            self.runs,
            self.interpolation_points,
            xss, yss, test_xss, test_yss, kgen,
            rgen, slfmgen, indepgen, {'verbosity': 100, 'min_grad_ratio': 0.2},
            max_procs=self.nthreads)
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
Suite.track_mean_time.benchmark_name = 'fx2007 time mean'
Suite.track_mean_time.units = 'seconds'
Suite.track_se_time.benchmark_name = 'fx2007 time standard error'
Suite.track_se_time.units = 'seconds'
Suite.track_mean_smse.benchmark_name = 'fx2007 smse mean'
Suite.track_se_smse.benchmark_name = 'fx2007 smse standard error'
Suite.track_mean_nlpd.benchmark_name = 'fx2007 nlpd mean'
Suite.track_se_nlpd.benchmark_name = 'fx2007 nlpd standard error'


def main():
    activate_logs()

    if is_validation():
        import runlmc.lmc.stochastic_deriv
        runlmc.lmc.stochastic_deriv.StochasticDeriv.N_IT = 1
        runs = 1
        cogp_runs = 1
        interpolation_points = 10
        inducing_points = 10
        nthreads = ''
    else:
        runs = 10
        cogp_runs = 10
        interpolation_points = None
        inducing_points = 100
        nthreads = 4

    llgp_stats = Suite(runs, interpolation_points, nthreads).setup_cache()
    print('---> llgp Q1R2 m', interpolation_points, statsline(llgp_stats))
    cogp_stats, _, _ = cogp_fx2007(cogp_runs,
                                   inducing_points,
                                   nthreads)
    print('---> cogp m', inducing_points, statsline(cogp_stats))
    latex_table('results_fx2007.tex',
                ['LLGP', 'COGP'],
                [llgp_stats, cogp_stats])


if __name__ == '__main__':
    main()
