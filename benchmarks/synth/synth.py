# compares to cogp on fake data

import os
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
from standard_tester import *
import pickle

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt


def load(f):
    with open('out/' + f + '.pkl', 'rb') as handle:
        return pickle.load(handle)


def dump(x, f):
    with open('out/' + f + '.pkl', 'wb') as handle:
        pickle.dump(x, handle, pickle.HIGHEST_PROTOCOL)


def main():
    np.random.seed(1234)
    activate_logs()

    llgp_runs = 3
    cogp_runs = 3
    cogp_max_it = 100 # like in COGP sarcos
    interpolating_points = [25, 25]
    nthreads = 16
    inducing_points = [500]
    nbatches = [1000]

    kgen, rgen, slfmgen, indepgen = synth_gen()
    xss, yss, test_xss, test_yss = synth()
    stats = bench_runlmc(llgp_runs, interpolating_points, xss, yss, test_xss,
                test_yss, kgen, rgen, slfmgen, indepgen, {}, max_procs=nthreads,
                tolerance=1e-3) # reduce tol a bit b/c of large problem size
    dump(stats, 'llgp_stats')
    # stats = load('llgp_stats')

    all_stats = [stats]
    colnames = ['LLGP']
    
    for ind, nb in zip(inducing_points, nbatches):
        cogp_stats = cogp_synth(
            cogp_runs, ind, nthreads, nb, cogp_max_it)
        dump(cogp_stats, 'cogp_stats-{}-{}'.format(ind, nb))
        # cogp_stats = load('cogp_stats-{}-{}'.format(ind, nb))   
        all_stats.append(cogp_stats)
        colnames.append(r'\begin{tabular}{c}LLGP\\$m=' +
                        str(ind) + ',n_b=' + str(nb) +
                        r'$\end{tabular}')

    latex_table('results_synth.tex', colnames, all_stats)

if __name__ == '__main__':
    main()
