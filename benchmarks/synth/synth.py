# compares to cogp on fake data

import os
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
from standard_tester import *
import pickle
                                                                                                                                                                                                              
import matplotlib as mpl                                                                                                                                                                                                         
mpl.use('Agg')                                                                                                                                                                                                                   
import matplotlib.pyplot as plt 

def dump(x, f):
    with open('out/' + f + '.pkl', 'wb') as handle:
        pickle.dump(x, handle, pickle.HIGHEST_PROTOCOL)

def main():
    np.random.seed(1234)
    activate_logs()

    max_it = 100
    interpolating_points = [50, 50]
    nthreads = 16
    inducing_points = 500 # like in COGP sarcos
    iters = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

    kgen, rgen, slfmgen, indepgen = synth_gen()
    
    times = []
    times, smses, nlpds = [], [], []
    lmc = None
    ctr = 0
    timer = contexttimer.Timer()
    xss, yss, test_xss, test_yss = synth()
    fk = FunctionalKernel(
        D=len(xss),
        lmc_kernels=kgen(),
        lmc_ranks=rgen(),
        slfm_kernels=slfmgen(),
        indep_gp=indepgen())

    def perf_cb():
        nonlocal times, smses, nlpds, lmc, ctr, timer
        ctr += 1
        if ctr % 10 != 0:
            return

        timer.__exit__(None, None, None)
        np.save(TMP + 'lmc-iter-{}'.format(ctr), lmc.param_array)
        pred_yss, pred_vss = lmc.predict(test_xss)
        smses.append(smse(test_yss, pred_yss, yss))
        nlpds.append(nlpd(test_yss, pred_yss, pred_vss))
        prev = times[-1] if times else 0
        times.append(timer.elapsed + prev)
        print('[llgp] time {:.0f} smse {:.2f} nlpd {:.2f}'.format(
            times[-1], smses[-1], nlpds[-1]))
        timer.__enter__()
        
    opt = AdaDelta(max_it=max_it, min_grad_ratio=0, callback=perf_cb, verbosity=0)
    timer.__enter__()
    lmc = InterpolatedLLGP(xss, yss, functional_kernel=fk,
                           normalize=True, m=interpolating_points,
                           max_procs = nthreads)
    lmc.optimize(optimizer=opt)
    timer.__exit__(None, None, None)
    
    for i in ['times', 'smses', 'nlpds']:
        dump(eval(i), 'llgp_' + i)

    cogp_times, cogp_smses, cogp_nlpds = [], [], []
    for it in iters:
        (ctime, _), (csmse, _), (cnlpd, _) = cogp_synth(1, inducing_points, nthreads, it)
        print('[cogp] time {:.0f} smse {:.2f} nlpd {:.2f}'.format(
            ctime, csmse, cnlpd))
        cogp_times.append(ctime)
        cogp_smses.append(csmse)
        cogp_nlpds.append(cnlpd)
            
    for i in ['times', 'smses', 'nlpds']:
        i = 'cogp_' + i
        dump(eval(i), i)
            
    figfile = 'out/learning-curves.pdf'
    print('writing output to', figfile)
    
    _, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 4))

    ax1.plot(times, smses, c='red', ls='-', marker='s')
    ax1.plot(cogp_times, cogp_smses, c='blue', ls='-', marker='^')
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('SMSE')
    ax1.set_title('SMSE vs training time')

    ax2.plot(times, nlpds, c='red', ls='-', marker='s')
    ax2.plot(cogp_times, cogp_nlpds, c='blue', ls='-', marker='^')
    ax2.set_xlabel('time (s)')
    ax2.set_ylabel('NLPD')
    ax2.set_title('NLPD vs training time')
            
    plt.savefig(figfile, format='pdf', bbox_inches='tight')

if __name__ == '__main__':
    main()
