# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import climin
from paramz.optimization import Optimizer
import numpy as np

# TODO: log instead of print

class AdaDelta(Optimizer):

    def __init__(self, **kwargs):
        super().__init__()
        default = {
            'step_rate':1, 'decay':0.5, 'momentum':0.5, 'offset':1e-4,
            'max_it':100, 'verbosity':0, 'min_grad':1e-3}
        default.update(**kwargs)
        self.kwargs = default
        print(default)

    def opt(self, x, f_fp=None, f=None, fp=None):
        exclude = {'verbosity', 'min_grad', 'max_it'}
        ada_kwargs = {k:v for k, v in self.kwargs.items() if k not in exclude}
        ada = climin.Adadelta(x, fp, **ada_kwargs)

        verbosity = self.kwargs['verbosity']
        max_it = self.kwargs['max_it']
        delta = max_it // verbosity if verbosity else 0
        gn_cut = self.kwargs['min_grad']

        if self.kwargs['verbosity']:
            print('starting AdaDelta', self.kwargs)

        info = None
        for info in ada:
            gn = np.linalg.norm(info['gradient'], np.inf)
            if verbosity and info['n_iter'] % delta == 0:
                print('iteration', info['n_iter'],
                      'grad inf-norm {:10.4e}'.format(gn))
            if info['n_iter'] >= max_it or gn < gn_cut:
                break

        if verbosity:
            print('finished adadelta optimization {:6d} iterations '
                  '{:10.4e} grad inf-norm'.format(
                      info['n_iter'],
                      np.linalg.norm(info['gradient'], np.inf)))
        self.x_opt = x
