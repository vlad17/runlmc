# compares to cogp on fake data

import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys
import numpy as np
from standard_tester import *
import pickle

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt


def load(f):
    with open(sys.argv[2] + os.path.sep + f + '.pkl', 'rb') as handle:
        return pickle.load(handle)


def dump(x, f):
    with open(sys.argv[2] + os.path.sep + f + '.pkl', 'wb') as handle:
        pickle.dump(x, handle, pickle.HIGHEST_PROTOCOL)


def main():
    np.random.seed(1234)
    activate_logs()

    llgp_runs = 3
    cogp_runs = 3
    cogp_max_it = [100, 1000]
    interpolating_points = [25, 25]
    nthreads = 16
    inducing_points = [500, 500]
    nbatches = [1000, 1000]
    cogp_colnames = ['COGP', 'COGP+']

    all_stats = []
    colnames = []

    if len(sys.argv) > 3:  # COGP grid-search mode
        nthreads = ''
        inducing_points = [int(sys.argv[3])]
        nbatches = [int(sys.argv[4])]
        cogp_max_it = [int(sys.argv[5])]
        cogp_runs = 1
        cogp_colnames = ['COGP']
    else:
        import h5py
        kgen, rgen, slfmgen, indepgen = synth_gen()
        xss, yss, test_xss, test_yss = synth()
        stats, lmc = bench_runlmc(llgp_runs, interpolating_points, xss, yss, test_xss,
                                  test_yss, kgen, rgen, slfmgen, indepgen, {}, max_procs=nthreads,
                                  # reduce tol a bit b/c of large problem size
                                  tolerance=1e-3, return_lmc=True)
        approx = lmc.kernel.K.as_numpy()
        exact = lmc.K()
        with h5py.File('llgp-mats.h5', 'w') as f:
            f.create_dataset('approx', data=approx)
            f.create_dataset('exact', data=exact)
        dump(stats, 'llgp_stats')
        #
        # stats = load('llgp_stats')
        all_stats = [stats]
        colnames = ['LLGP']

    for ind, nb, mi, name in zip(inducing_points, nbatches, cogp_max_it, cogp_colnames):
        cogp_stats = cogp_synth(
            cogp_runs, ind, nthreads, nb, mi)
        dump(cogp_stats, 'cogp_stats-{}-{}'.format(ind, nb))
        # cogp_stats = load('cogp_stats-{}-{}'.format(ind, nb))
        all_stats.append(cogp_stats)
        colnames.append(name)

    latex_table('results_synth.tex', colnames, all_stats)


if __name__ == '__main__':
    main()
