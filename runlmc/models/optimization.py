# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import climin
from paramz.optimization import Optimizer
import numpy as np

from .lmc import LMC

# TODO(cleanup) move to numpy convenience
# TODO(test)
class MovingAverage:
    def __init__(self, n):
        self.n = n
        self.i = 0
        self.past = np.zeros(n)
        self.total = 0

    def push(self, x):
        update_ix = self.i % self.n
        self.total -= self.past[update_ix]
        self.total += x
        self.past[update_ix] = x
        self.i += 1
        if self.i < self.n:
            return self.total / self.i
        return self.total / self.n

# TODO(test)
class AdaDelta(Optimizer):

    def __init__(self, **kwargs):
        super().__init__()
        default = {
            'step_rate':1, 'decay':0.9, 'momentum':0.5, 'offset':1e-4,
            'max_it':100, 'verbosity':0, 'min_grad_ratio':0.5, 'roll':1,
            'permitted_drops':None}
        default.update(**kwargs)
        if default['permitted_drops'] is None:
            default['permitted_drops'] = max(default['max_it'] // 20, 1)
        self.kwargs = default

    def opt(self, x, f_fp=None, f=None, fp=None):
        exclude = ['verbosity', 'min_grad_ratio', 'max_it', 'roll',
                   'permitted_drops']
        ada_kwargs = {k:v for k, v in self.kwargs.items() if k not in exclude}
        ada = climin.Adadelta(x, fp, **ada_kwargs)

        verbosity, min_grad_ratio, max_it, roll, permitted_drops = (
            self.kwargs[x] for x in exclude)
        delta = max(max_it // verbosity, 1) if verbosity else 0
        rolling_max = 1e-10
        rolling_ave = MovingAverage(roll)

        if self.kwargs['verbosity']:
            print('starting adadelta', self.kwargs)

        info = None
        for info in ada:
            gn = np.linalg.norm(info['gradient'], LMC.EVAL_NORM)
            ave = rolling_ave.push(gn)
            rolling_max = max(ave, rolling_max)

            if verbosity and info['n_iter'] % delta == 0:
                print('iteration {:8d}'.format(info['n_iter']),
                      'grad norm {:10.4e}'.format(gn))

            if ave < min_grad_ratio * rolling_max:
                permitted_drops -= 1
                if permitted_drops <= 0:
                    break

            if info['n_iter'] >= max_it:
                break

        if verbosity:
            print('finished adadelta optimization\n'
                  '    {:10d} iterations\n'
                  '    {:10.4e} final grad norm\n'
                  '    {:10.4e} final MA({}) grad norm\n'
                  '    {:10.4e} max MA({}) grad norm\n'
                  '    norm used {}'
                  .format(info['n_iter'], gn, ave, roll,
                          rolling_max, roll, LMC.EVAL_NORM))
        self.x_opt = x
