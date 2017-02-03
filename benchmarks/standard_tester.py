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

from runlmc.models.lmc import LMC
from runlmc.kern.rbf import RBF
from runlmc.models.optimization import AdaDelta
from runlmc.models.gpy_lmc import GPyLMC

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
    fx2007_train = fx2007.copy()
    # they did the experiment in R...
    holdout = {'CAD': slice(49, 99),
               'JPY': slice(99, 149),
               'AUD': slice(149, 199)}
    for col in fx2007.columns:
        if col not in holdout:
            holdout[col] = slice(0, 0)
    holdin = {col: np.r_[0:ixs.start, ixs.stop:len(fx2007)]
              for col, ixs in holdout.items()}

    xss = []
    yss = []
    all_ixs = np.arange(len(fx2007))
    for col in fx2007.columns:
        select = np.ones(len(fx2007), dtype=bool)
        select[holdout[col]] = False
        select[fx2007[col].isnull().values] = False
        xss.append(all_ixs[select])
        currency2usd = fx2007_train[col][select].values
        # Don't ask me, this was in the Nguyen 2014 paper code
        usd2currency = np.reciprocal(currency2usd)
        yss.append(usd2currency)

    test_fx = ['CAD', 'JPY', 'AUD']

    test_xss = [all_ixs[holdout[col]] for col in fx2007.columns]
    test_yss = [np.reciprocal(fx2007.ix[holdout[col], col]) for col in fx2007.columns]
    return xss, yss, test_xss, test_yss, fx2007, all_ixs, holdout, holdin

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
    return np.mean([0.5 * np.mean(
            np.square(test_ys - pred_ys) / pred_vs
            + np.log(2 * np.pi * pred_vs))
            for test_ys, pred_ys, pred_vs in
            zip(test_yss, pred_yss, pred_vss)])

def runlmc(num_runs, m, xss, yss, test_xss, test_yss,
           kerns, ranks, optimizer_opts):
    times, smses, nlpds, lls = [], [], [], []
    best = None
    for _ in range(num_runs):
        lmc = LMC(xss, yss, kernels=kerns, ranks=ranks, normalize=True, m=m)
        opt = AdaDelta(**optimizer_opts)
        with contexttimer.Timer() as t:
            lmc.optimize(optimizer=opt)
        times.append(t.elapsed)
        lls.append(lmc.log_likelihood())
        pred_yss, pred_vss = lmc.predict(test_xss)
        smses.append(smse(test_yss, pred_yss, yss))
        nlpds.append(nlpd(test_yss, pred_yss, pred_vss))
        if min(lls) == lls[-1]:
            best = lmc
    return np.mean(times), np.mean(smses), np.mean(nlpds), best

def dtcvar(num_runs, m, xss, yss, test_xss, test_yss,
           kerns, ranks, optimizer_opts):
    times, smses, nlpds, lls = [], [], [], []
    best = None
    for _ in range(num_runs):
        dtcvar = GPyLMC(xss, yss, kernels=ks, ranks=ranks, sparse=num_inducing)
        with contexttimer.Timer() as t:
            dtcvar.optimize(**optimizer_opts)
        times.append(t.elapsed)
        pred_yss, pred_vss = dtcvar.predict(test_xss)
        smses.append(smse(test_yss, pred_yss, yss))
        nlpds.append(nlpd(test_yss, pred_yss, pred_vss))
    return np.mean(times), np.mean(smses), np.mean(nlpds), best

def cogp_fx2007(num_runs, num_inducing):
    # Download paper code if it is not there
    if not os.path.isdir('/tmp/cogp'):
        print('cloning COGP repo')
        subprocess.call(['git', 'clone', 'git@github.com:vlad17/cogp.git', '/tmp/cogp'])
    if not os.path.isfile('/tmp/fx2007.csv'):
        print('exporting fx2007 to csv')
        fx2007.fillna(-1).to_csv('/tmp/fx2007.csv', header=False, index=False)
    # This runs the COGP code; only learning is timed
    cmd = ['matlab', '-nojvm', '-r',
           """infile='/tmp/fx2007.csv';M={};runs={};cogp_fx2007;exit"""
           .format(num_inducing, num_runs)]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        cwd=os.getcwd())
    mout = process.communicate()[0]
    with open('/tmp/out-{}-{}'.format(num_runs, num_inducing), 'w') as f:
        f.write(mout)
    ending = mout[mout.find('mean times'):]
    time = float(re.match('\D*([-+e\.\d]*)', ending).groups()[0])
    ending = ending[ending.find('mean smses'):]
    smse = float(re.match('\D*([-+e\.\d]*)', ending).groups()[0])
    ending = ending[ending.find('mean nlpds'):]
    nlpd = float(re.match('\D*([-+e\.\d]*)', ending).groups()[0])
    return time, smse, nlpd
