[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

# runlmc

Do you like to apply Bayesian nonparameteric methods to your regressions? Are you frequently tempted by the flexibility that kernel-based learning provides? Do you have trouble getting structured kernel interpolation or various training conditional inducing point approaches to work in a non-stationary multi-output setting?

If so, this package is for you.

**runlmc** is a Python 3 package designed to extend structural efficiencies from _Scalable inference for structured Gaussian process models_ (Staaçi 2012) and _Thoughts on Massively Scalable Gaussian Processes_ (Wilson et al 2015) to the non-stationary setting of linearly coregionalized multiple-output regressions. For the single output setting, MATLAB implementations are available [here](http://www.gaussianprocess.org/gpml/code/matlab/doc/).

In other words, this provides a matrix-free implementation of multi-output GPs for certain covariances. As far as I know, this is also the only matrix-free implementation for single-output GPs in python.

## Usage Notes

* Currently, I'm only supporting 1 input dimension.
* Docs avaliable [on the product page](https://vlad17.github.io/runlmc)

## A note on GPy

[GPy](https://github.com/SheffieldML/GPy) is a way more general GP library that was a strong influence in the development of this one. I've tried to stay as faithful as possible to its structure.

I've re-used a lot of the GPy code. The main issue with simply adding my methods to GPy is that the API used to interact between GPy's `kern`, `likelihood`, and `inference` packages centers around the `dL_dK` object, a matrix derivative of the likelihood with respect to covariance. The materialization of this matrix is the very thing my algorithm tries to avoid for performance.

If there is some quantifiable success with this approach then integration with GPy would be a reasonable next-step.

## Examples and Benchmarks

### Snippet

Note: currently derivatives are not implemented, nor is the package optimized for speed yet.

Full example 

    n_per_output = [65, 100]
    xss = list(map(np.random.rand, n_per_output))
    yss = [f(2 * np.pi * xs) + np.random.randn(len(xs)) * 0.05
           for f, xs in zip([np.sin, np.cos], xss)]
    ks = [RBF(name='rbf{}'.format(i)) for i in range(nout)]
    lmc = LMC(xss, yss, kernels=ks)
    # ... plotting code
        
![unopt](https://raw.githubusercontent.com/vlad17/runlmc/master/examples/unopt.png)

    lmc.optimize(optimizer=AdaDelta()) # batteries included!
    # ... more plotting code
    
![opt](https://raw.githubusercontent.com/vlad17/runlmc/master/examples/opt.png)
        
### Running the Examples and Benchmarks

Make sure that the directory root is in the `PYTHONPATH` when running the benchmarks. E.g., from the directory root:

    PYTHONPATH=.. jupyter notebook examples/example.ipynb
    PYTHONPATH=. python benchmarks/bench.py
    cd benchmarks && OMP_NUM_THREADS=1 PYTHONPATH=.:.. fx2007.py
    
## Dev Stuff

Required packages for running (Python 3 versions): `numpy scipy climin GPy paramz contexttimer pandas`. For dev `nose sphinx`.

All below invocations should be done from the repo root.
 
| Command           | Purpose  |
| ----------------- | -------- |
| `./style.sh`      | Check style with pylint, ignoring TODOs and locally-disabled warnings. |
| `./docbuild.sh`   | Regenerate docs (index will be in `doc/_generated/_build/runlmc.html`) |
| `./docpush.sh`   | Publish current docs (requries write access to repo) |
| `nosetests -l DEBUG`       | Run unit tests |

### Roadmap

0. code/run large benchmarks (no ipynb in benchmarks). -> benchmarks should regen kernels.
0. Write up ICML paper.
0. rm 33k stuff from benchmarks
0. Clean up benchmarks: set tasks for general cleanliness, figure out what to do with standard_tester.py, remove path dependency (right now access `../data` requires you to be in the benchmarks/ directory for invocations). How to run - info should go in their main pages, README should say how to print the help paragraph for all the scripts. Move slurm script for large kernel perf grid into repo, clean it up too. clean up OMP_NUM_THREADS stuff too.
0. Coalesce pool (rm extrapool): Starmapper multiprocessing interface - pool-based, within-thread. PoolGenerator - make on-the-fly pools (generator should cleanup on failure), with closing() and cleanup. Use is `self.pool_generator.make(par) as pool:`. If in LMC pool=... specified, then wrap it in a dummy generator that returns the same pool always. If pool=... is unspecified, then pool_generator will make pools on the fly with desired parallelism min with cpu count (but will short-circuit to blocking within-thread parallelism below size threshold/cpu_count=1). no chunks. [bugfix pool](https://github.com/python/cpython/pull/57/files). Update benchmarks to new interface.
0. Generic benchmarks-run-all script.
0. TODO(general-solve) Preconditioner?
    * Does chan Preconditioner carry over to SKI approximation? -> no, it doesn't
    * Do other inner circulant preconditioners (e.g., whittle) help inversion? -> no
    * Cache LCG solutions over iterations?
    * Cutajar 2016 iterative inversion approach?
0. Minor perf improvements: what helps?
    * CPython; numba.
    * In-place multiplication where possible
    * square matrix optimizations
    * TODO(sparse-derivatives)
0. travis-ci, auto doc builds, auto benchmarks
0. TODO(sum-fast) low-rank dense multiplications give SumKernel speedups?
0. multidimensional inputs and ARD.
0. TODO(prior). Compare to [spike and slab](http://www.aueb.gr/users/mtitsias/publications.html), also try MedGP (e.g., three-parameter beta) - add tests for priored versions of classes, some tests in parameterization/ (priors should be value-cached, try to use an external package)

### Considerations 

* Real datasets: [link1](http://www.robots.ox.ac.uk/~davidc/publications_MTGP.php) 
* Consider other approximate inverse algorithms: see Thm 2.4 of [Agarwal, Allen-Zhu, Bullins, Hazan, Ma 2016](https://arxiv.org/abs/1611.01146)
0. Logdet Approximations? (1) [Chebyshev-Hutchinson](https://arxiv.org/abs/1503.06394) [Code](https://sites.google.com/site/mijirim/logdet) (2) [Integral Probing](https://arxiv.org/abs/1504.02661).

### Low-priority Tasks

0. TODO(cleanup) - apprx to approx everywhere
0. Allow extrapolation in util.interpolation.py
0. TODO(test) - document everything that's missing documentation along the way.
0. test `LMC._raw_predict` unit testing, by using K_SKI() and anlogous math
0. np.linalg.eigvalsh -> scipy.linalg.eigvalsh (numpy.linalg -> scipy.linalg as la, scipy.sparse.linalg as sla)
0. rename `rand_psd` -> `rand_pd`
0. Continuous integration for unit tests
0. mean functions
0. product kernels (multiple factors) 
0. active dimension optimization
