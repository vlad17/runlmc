# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

# pylint: skip-file

import subprocess
import re
import os

import scipy.io as sio
import contexttimer
import numpy as np
import pandas as pd
import numpy as np

from runlmc.models.lmc import LMC
from runlmc.kern.rbf import RBF
from runlmc.models.optimization import AdaDelta
from runlmc.models.gpy_lmc import GPyLMC
from runlmc.util.numpy_convenience import begin_end_indices

import tempfile

TMP = tempfile.gettempdir()

def _foreign_exchange_shared():
    # Adapts the foreign currency exchange problem
    # Collaborative Multi-output Gaussian Processes
    # Nguyen and Bonilla et al. 2014

    fx_files = ['2007-2009.csv', '2010-2013.csv', '2014-2017.csv']
    fx = pd.concat([pd.read_csv('../data/fx/{}'.format(f), index_col=1) for f in fx_files])
    fx.drop(['Wdy', 'Jul.Day'], axis=1, inplace=True)
    fx.rename(columns={old: old[:3] for old in fx.columns}, inplace=True)
    return fx

def foreign_exchange_2007():
    # This example uses only 2007 data
    fx = _foreign_exchange_shared()
    fx2007 = fx.ix['2007/01/01':'2008/01/01']
    # they did the experiment in R...
    holdout = {'CAD': slice(49, 99),
               'JPY': slice(99, 149),
               'AUD': slice(149, 199)}
    for col in fx2007.columns:
        if col not in holdout:
            holdout[col] = slice(0, 0)
    holdin = {}
    for col in fx2007.columns:
        select = np.ones(len(fx2007), dtype=bool)
        select[fx2007[col].isnull().values] = False
        select[holdout[col]] = False
        holdin[col] = np.r_[np.flatnonzero(select)]

    xss = []
    yss = []
    all_ixs = np.arange(len(fx2007))
    for col in fx2007.columns:
        xss.append(all_ixs[holdin[col]])
        currency2usd = fx2007[col][holdin[col]].values
        # Don't ask me, this was in the Nguyen 2014 paper code
        usd2currency = np.reciprocal(currency2usd)
        yss.append(usd2currency)

    test_fx = ['CAD', 'JPY', 'AUD']
    test_xss = [all_ixs[holdout[col]] for col in fx2007.columns]
    test_yss = [np.reciprocal(fx2007.ix[holdout[col], col])
                for col in fx2007.columns]
    return xss, yss, test_xss, test_yss, test_fx, fx2007.columns

def weather():
    sensors = ['bra', 'cam', 'chi', 'sot']
    expected_nas = [100, 0, 15, 1002]
    holdout = [None, (10.2, 10.8), (13.5, 14.2), None]
    xss, yss = [], []
    test_xss, test_yss = [], []
    for sensor, expected_na, hold in zip(sensors, expected_nas, holdout):
        y = pd.read_csv('../data/weather/{}y.csv'.format(sensor),
                        header=None, names=['WSPD','WD','GST','ATMP'],
                        usecols=['ATMP'])
        x = pd.read_csv('../data/weather/{}x.csv'.format(sensor),
                        header=None, names=['time'])
        assert (x['time'] == -1).sum() == 0
        assert (y['ATMP'] == -1).sum() == expected_na
        y['ATMP'][y['ATMP'] == -1] = np.nan
        y.dropna(inplace=True)
        xy = pd.concat([x, y], axis=1, join='inner')
        if hold is None:
            test_xss.append(np.array([]))
            test_yss.append(np.array([]))
            xss.append(xy['time'].values)
            yss.append(xy['ATMP'].values)
        else:
            sel = xy['time'].between(hold[0], hold[1])
            test_xss.append(xy.loc[sel]['time'].values)
            test_yss.append(xy.loc[sel]['ATMP'].values)
            xss.append(xy.loc[~sel]['time'].values)
            yss.append(xy.loc[~sel]['ATMP'].values)

    return xss, yss, test_xss, test_yss, sensors

def toy_sinusoid():
    # Adapts the 2-output toy problem from
    # Collaborative Multi-output Gaussian Processes
    # Nguyen and Bonilla et al. 2014

    # Their example uses a grid of inputs. To make it harder (for runlmc)
    # we instead look at uniformly distributed inputs.

    sz = 1500
    xss = [np.random.uniform(-10,10,size=sz) for _ in range(2)]
    f1 = lambda x: np.sin(x) + 1e-7 + np.random.randn(len(x)) * 1e-2
    f2 = lambda x: -np.sin(x) + 1e-7 + np.random.randn(len(x)) * 1e-2
    yss = [f1(xss[0]), f2(xss[1])]
    ks = [RBF(name='rbf0')]
    ranks=[1]
    pred_xss = [np.linspace(-11, 11, 100) for _ in range(2)]
    test_yss = [f1(pred_xss[0]), f2(pred_xss[1])]
    return None

def filter_list(ls, ixs):
    return [ls[i] for i in ixs]

def filter_nonempty_cols(a, b, c):
    nonempty_ixs = [i for i, x in enumerate(a) if len(x) > 0]
    return (filter_list(x, nonempty_ixs) for x in (a, b, c))

def filter_by_selection(a, b, c, selects):
    return ([x[select] for x, select in zip(xs, selects)] for xs in (a, b, c))

def smse(test_yss, pred_yss, train_yss):
    test_yss, pred_yss, train_yss = filter_nonempty_cols(
        test_yss, pred_yss, train_yss)
    return np.mean([np.square(test_ys - pred_ys).mean() /
            np.square(train_ys.mean() - test_ys).mean() for
            test_ys, pred_ys, train_ys in
            zip(test_yss, pred_yss, train_yss)])

def nlpd(test_yss, pred_yss, pred_vss):
    test_yss, pred_yss, pred_vss = filter_nonempty_cols(
        test_yss, pred_yss, pred_vss)
    selectors = []
    skipped = 0
    for i, vs in enumerate(pred_vss):
        selectors.append(np.flatnonzero(vs))
        skipped += len(vs) - len(selectors[-1])
    if skipped > 0:
        print('warning: found {} of {} predictive variances set to 0'
              .format(skipped, sum(map(len, pred_vss))))
    test_yss, pred_yss, pred_vss = filter_by_selection(
        test_yss, pred_yss, pred_vss, selectors)
    test_yss, pred_yss, pred_vss = filter_nonempty_cols(
        test_yss, pred_yss, pred_vss)
    return np.mean([0.5 * np.mean(
            np.square(test_ys - pred_ys) / pred_vs
            + np.log(2 * np.pi * pred_vs))
            for test_ys, pred_ys, pred_vs in
            zip(test_yss, pred_yss, pred_vss)])

def runlmc(num_runs, m, xss, yss, test_xss, test_yss,
           kgen, rgen, slfmgen, indepgen, optimizer_opts, **kwargs):
    times, smses, nlpds = [], [], []
    for i in range(num_runs):
        ks = kgen()
        rs = rgen()
        slfm = slfmgen()
        indep = indepgen()
        lmc = LMC(xss, yss, kernels=ks, ranks=rs,
                  slfm_kerns=slfm, indep_gp=indep,
                  normalize=True, m=m, **kwargs)
        for i in range(lmc.nkernels['lmc']):
            print('LMC kernel', i, 'A matrix')
            print(eval('lmc.a{}'.format(i)).values)
            print('LMC kernel', i, 'kappa diag')
            print(eval('lmc.kappa{}'.format(i)).values)
        for i in range(lmc.nkernels['slfm']):
            i += lmc.nkernels['lmc']
            print('SLFM kernel', i, 'A matrix')
            print(eval('lmc.a{}'.format(i)).values)
        opt = AdaDelta(**optimizer_opts)
        with contexttimer.Timer() as t:
            lmc.optimize(optimizer=opt)
        times.append(t.elapsed)
        np.save(TMP + 'lmc-m{}-{}of{}-{}.npy'.format(m, i, num_runs, sum(map(len, xss))),
                lmc.param_array)
        pred_yss, pred_vss = lmc.predict(test_xss)
        smses.append(smse(test_yss, pred_yss, yss))
        nlpds.append(nlpd(test_yss, pred_yss, pred_vss))
        print('time', times[-1], 'smse', smses[-1], 'nlpd', nlpds[-1])
        last = lmc
    return times, smses, nlpds, last

def _download_cogp():
    # Download paper code if it is not there
    if not os.path.isdir(TMP + '/cogp'):
        print('cloning COGP repo')
        repo = 'git@github.com:vlad17/cogp.git'
        subprocess.call(['git', 'clone', repo, TMP + '/cogp'])

def env_no_omp():
    env = os.environ.copy()
    if 'OMP_NUM_THREADS' in env:
        del env['OMP_NUM_THREADS']
    return env

def cogp_fx2007(num_runs, inducing_pts):
    _download_cogp()
    datafile = '../data/fx/fx2007_matlab.csv'
    assert os.path.isfile(datafile)
    # This runs the COGP code; only learning is timed
    cmd = ['matlab', '-nojvm', '-r',
           """infile='{}';M={};runs={};cogp_fx2007;exit"""
           .format(datafile, inducing_pts, num_runs)]
    with open(TMP + '/out-{}'.format(num_runs), 'w') as f:
        f.write(' '.join(cmd))
    benchmark_dir = os.getcwd() + '/../benchmarks'
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        cwd=benchmark_dir,
        env=env_no_omp())
    mout = process.communicate()[0]
    with open(TMP + '/out-{}'.format(num_runs), 'a') as f:
        f.write(mout)
    ending = mout[mout.find('mean times'):]
    time = float(re.match('\D*([-+e\.\d]*)', ending).groups()[0])
    ending = ending[ending.find('mean smses'):]
    smse = float(re.match('\D*([-+e\.\d]*)', ending).groups()[0])
    ending = ending[ending.find('mean nlpds'):]
    nlpd = float(re.match('\D*([-+e\.\d]*)', ending).groups()[0])

    # the matlab script writes to this file
    test_fx = ['CAD', 'JPY', 'AUD']
    cogp_mu = pd.read_csv(TMP + '/cogp-fx2007-mu', header=None, names=test_fx)
    cogp_var = pd.read_csv(TMP + '/cogp-fx2007-var', header=None, names=test_fx)

    return time, smse, nlpd, cogp_mu, cogp_var

def cogp_weather(num_runs, M):
    _download_cogp()
    datafile = '../data/weather/'
    # This runs the COGP code; only learning is timed
    cmd = ['matlab', '-nojvm', '-r',
           """datadir='{}';M={};runs={};cogp_weather;exit"""
           .format(datafile, M, num_runs)]
    with open(TMP + '/outw-{}-{}'.format(num_runs, M), 'w') as f:
        f.write(' '.join(cmd))
    benchmark_dir = os.getcwd() + '/../benchmarks'
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        cwd=benchmark_dir,
        env=env_no_omp())
    mout = process.communicate()[0]
    with open(TMP + '/outw-{}-{}'.format(num_runs, M), 'a') as f:
        f.write(mout)
    ending = mout[mout.find('mean times'):]
    time = float(re.match('\D*([-+e\.\d]*)', ending).groups()[0])
    ending = ending[ending.find('mean smses'):]
    smse = float(re.match('\D*([-+e\.\d]*)', ending).groups()[0])
    ending = ending[ending.find('mean nlpds'):]
    nlpd = float(re.match('\D*([-+e\.\d]*)', ending).groups()[0])

    # the matlab script writes to this file
    test_fx = ['cam', 'chi']
    cogp_mu = pd.read_csv(TMP + '/cogp-weather-mu{}{}'.format(num_runs, M), header=None, names=test_fx)
    cogp_var = pd.read_csv(TMP + '/cogp-weather-var{}{}'.format(num_runs, M))

    return time, smse, nlpd, cogp_mu, cogp_var

def statprint(x):
    return '{:10.4f} ({:10.4f})'.format(np.mean(x), np.std(x))
