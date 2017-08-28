import pandas as pd
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

df = pd.read_csv('extracted_summary.csv', header=0).dropna()
print('num rows', len(df))
print('cols', list(df.columns))


def printimg(col, ttl, negate=False):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
    kernels = df['k'].unique()

    mult = -1 if negate else 1

    smax = 0
    smin = 0
    for ax, k in zip(axes.flat, kernels):
        konly = df[df['k'] == k]
        konly.set_index(['q', 'eps'], inplace=True)
        M = mult * np.log(konly.unstack()[col].values)
        smax = max(M.max(), smax)
        smin = min(M.min(), smin)

    im = None
    for ax, k in zip(axes.flat, kernels):
        konly = df[df['k'] == k]
        qs = konly['q']
        eps = konly['eps']
        konly.set_index(['q', 'eps'], inplace=True)
        M = mult * np.log(konly.unstack()[col].values)
        q_coord, eps_coord = konly.index.levels
        csort = np.argsort(-np.log(eps_coord))
        rsort = np.argsort(-q_coord)
        M = M[rsort][:, csort]
        im = ax.imshow(M, interpolation='nearest', cmap=plt.get_cmap('bwr'),
                       vmin=smin, vmax=smax)
        ax.set_xticks(np.arange(len(eps_coord)))
        ax.set_xticklabels(np.log10(eps_coord)[csort])
        ax.set_yticks(np.arange(len(q_coord)))
        ax.set_yticklabels(q_coord[rsort])
        for extreme in [M.argmin(), M.argmax()]:
            extreme = np.unravel_index(extreme, M.shape)
            ax.text(extreme[1], extreme[0], '{:.2f}'.format(
                M[extreme]), va='center', ha='center')
        ax.set_title(k, fontsize=14)

    if im:
        fig.text(0.51, 0.025, r'$\log_{10}\epsilon$', ha='center', fontsize=18)
        fig.text(0, 0.5, r'$Q$', va='center', rotation='vertical', fontsize=18)
        fig.subplots_adjust(right=0.7, top=1.1, hspace=0.4)
        plt.tight_layout()
        cbar_ax = fig.add_axes([1, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        f = col + '.eps'
        fig.suptitle(ttl, fontsize=18)
        plt.savefig(f, format='eps', bbox_inches='tight')
        print('made', f)

    plt.clf()


printimg('time_ratio', r'log ratio in cholesky to LLGP gradient runtime')
printimg('relgrad_l1',
         r'negative log relative error in $\|\nabla\mathcal{L}\|_1$', True)
printimg('relgrad_l2',
         r'negative log relative error in $\|\nabla\mathcal{L}\|_2$', True)
printimg('relalpha_l1',
         r'negative log relative error in $\|K^{-1}\mathbf{y}\|_1$', True)
printimg('relalpha_l2',
         r'negative log relative error in $\|K^{-1}\mathbf{y}\|_2$', True)
