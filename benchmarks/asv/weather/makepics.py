if __name__ == '__main__':
    import numpy as np
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt

    import subprocess

    cmd = ['grep', '-e', '--->', 'benchmarks/weather-out/stdout-weather.txt']
    process = subprocess.Popen(cmd,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               universal_newlines=True)
    out, err = process.communicate()
    print(err)

    lines = out.split('\n')

    llgp = [line.split() for line in lines]
    llgp = [line for line in llgp if line]
    llgp, cogp = llgp[:-1], llgp[-1]

    ms = [int(line[4]) for line in llgp]
    times = [float(line[6]) for line in llgp]
    se_times = [float(line[8][:-1]) for line in llgp]
    smses = [float(line[10]) for line in llgp]
    nlpds = [float(line[14]) for line in llgp]
    se_nlpds = [float(line[16][:-1]) for line in llgp]

    fig, ax1 = plt.subplots()
    ax1.errorbar(ms, nlpds, yerr=se_nlpds, c='r', marker='s', ecolor='pink')
    ax1.set_ylabel('NLPD', color='r')
    ax1.set_xlabel(r'interpolating points $m$')
    ax2 = ax1.twinx()
    jittered_ms = [m - 10 for m in ms]
    ax2.errorbar(jittered_ms, times, yerr=se_times, c='b', marker='o', ls=':', ecolor='lightblue')
    ax2.set_ylabel('runtime (s)', color='b')
    ax2.set_xlim([min(ms) - 30, max(ms) + 20])

    fig.tight_layout()
    plt.savefig('benchmarks/weather-out/m_time_nlpd.eps', format='eps', bbox_inches='tight')

    print('saved m_time_nlpd.eps')
