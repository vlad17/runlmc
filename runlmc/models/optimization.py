# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import climin
from paramz.optimization import Optimizer
import scipy.linalg as la

from .interpolated_llgp import InterpolatedLLGP

# TODO(test)


class AdaDelta(Optimizer):

    @staticmethod
    def noop():
        pass

    def __init__(self, **kwargs):
        super().__init__()
        default = {
            'step_rate': 1, 'decay': 0.9, 'momentum': 0.5, 'offset': 1e-4,
            'max_it': 100, 'verbosity': 0, 'min_grad_ratio': 0.1,
            'permitted_drops': 5, 'callback': AdaDelta.noop}
        default.update(**kwargs)
        self.kwargs = default

    def _prepare_adadelta(self, x, fp):
        exclude = ['verbosity', 'min_grad_ratio', 'max_it',
                   'permitted_drops', 'callback']
        ada_kwargs = {k: v for k, v in self.kwargs.items() if k not in exclude}
        return climin.Adadelta(x, fp, **ada_kwargs)

    def _grad_norm(self, info):
        return la.norm(info['gradient'], InterpolatedLLGP.EVAL_NORM)

    def _print_prolog(self):
        if self.kwargs['verbosity']:
            print('starting adadelta', self.kwargs)

    def _print_epilog(self, info):
        if self.kwargs['verbosity']:
            grad_norm = self._grad_norm(info)
            print('finished adadelta optimization\n'
                  '    {:10d} iterations\n'
                  '    {:10.4e} final grad norm\n'
                  '    norm used {}'
                  .format(info['n_iter'],
                          grad_norm,
                          InterpolatedLLGP.EVAL_NORM))

    def _print_update(self, info, grad_norm):
        max_it, verbosity = (self.kwargs[x] for x in ['max_it', 'verbosity'])
        printing_delta = max(max_it // verbosity, 1) if verbosity else 0
        if verbosity and info['n_iter'] % printing_delta == 0:
            print('iteration {:8d}'.format(info['n_iter']),
                  'grad norm {:10.4e}'.format(grad_norm))

    def opt(self, x, f_fp=None, f=None, fp=None):
        ada = self._prepare_adadelta(x, fp)

        min_grad_ratio = self.kwargs['min_grad_ratio']
        permitted_drops = self.kwargs['permitted_drops']
        max_it = self.kwargs['max_it']
        rolling_max = 0

        self._print_prolog()

        for info in ada:
            grad_norm = self._grad_norm(info)
            rolling_max = max(grad_norm, rolling_max)

            self._print_update(info, grad_norm)
            self.kwargs['callback']()

            if grad_norm < min_grad_ratio * rolling_max:
                permitted_drops -= 1

            if info['n_iter'] >= max_it or permitted_drops <= 0:
                self._print_epilog(info)
                break

        self.x_opt = x
