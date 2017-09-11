[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Documentation Status](https://readthedocs.org/projects/runlmc/badge/?version=latest)](http://runlmc.readthedocs.io/en/latest/?badge=latest)
[![CI](https://api.travis-ci.org/vlad17/runlmc.svg?branch=master)](https://travis-ci.org/vlad17/runlmc)
[![codecov](https://codecov.io/gh/vlad17/runlmc/branch/master/graph/badge.svg)](https://codecov.io/gh/vlad17/runlmc)
[![asv](http://img.shields.io/badge/benchmarked%20by-asv-green.svg?style=flat)](https://vlad17.github.io/runlmc/)


# runlmc

Do you like to apply Bayesian nonparameteric methods to your regressions? Are you frequently tempted by the flexibility that kernel-based learning provides? Do you have trouble getting structured kernel interpolation or various training conditional inducing point approaches to work in a non-stationary multi-output setting?

If so, this package is for you.

**runlmc** is a Python 3.4+ package designed to extend structural efficiencies from _Scalable inference for structured Gaussian process models_ (Staaçi 2012) and _Thoughts on Massively Scalable Gaussian Processes_ (Wilson et al 2015) to the non-stationary setting of linearly coregionalized multiple-output regressions. For the single output setting, MATLAB implementations are available [here](http://www.gaussianprocess.org/gpml/code/matlab/doc/).

In other words, this provides a matrix-free implementation of multi-output GPs for certain covariances. As far as I know, this is also the only matrix-free implementation for single-output GPs in python.

## Usage Notes

* Currently, I'm only supporting 1 input dimension.
* Zero-mean only for now.
* Check out the [latest documentation](http://runlmc.readthedocs.io/en/latest/)

## A note on GPy

[GPy](https://github.com/SheffieldML/GPy) is a way more general GP library that was a strong influence in the development of this one. I've tried to stay as faithful as possible to its structure.

I've re-used a lot of the GPy code. The main issue with simply adding my methods to GPy is that the API used to interact between GPy's `kern`, `likelihood`, and `inference` packages centers around the `dL_dK` object, a matrix derivative of the likelihood with respect to covariance. The materialization of this matrix is the very thing my algorithm tries to avoid for performance.

If there is some quantifiable success with this approach then integration with GPy would be a reasonable next-step.

## Examples and Benchmarks

### Snippet

    n_per_output = [65, 100]
    xss = list(map(np.random.rand, n_per_output))
    yss = [f(2 * np.pi * xs) + np.random.randn(len(xs)) * 0.05
           for f, xs in zip([np.sin, np.cos], xss)]
    ks = [RBF(name='rbf{}'.format(i)) for i in range(nout)]
    lmc = LMC(xss, yss, kernels=ks)
    # ... plotting code
        
![unopt](https://raw.githubusercontent.com/vlad17/runlmc/master/examples/unopt.png)

    lmc.optimize()
    # ... more plotting code
    
![opt](https://raw.githubusercontent.com/vlad17/runlmc/master/examples/opt.png)

For runnable code, check `examples/`.
        
### Running the Examples and Benchmarks

Make sure that the directory root is in the `PYTHONPATH` when running the benchmarks. E.g., from the directory root:

    PYTHONPATH=.. jupyter notebook examples/example.ipynb
    cd benchmarks/fx2007 && ./run.sh # will take a while!
    
## Dev Stuff

Required packages for running (Python 3): `pip install -r requirements.txt`.

All below invocations should be done from the repo root.
 
| Command           | Purpose  |
| ----------------- | -------- |
| `./style.sh`      | Check style with pylint, ignoring TODOs and locally-disabled warnings. |
| `./docbuild.sh`   | Regenerate docs (index will be in `doc/_generated/_build/index.html`) |
| `nosetests`       | Run unit tests |
| `./arxiv-tar.sh`       | Create an arxiv-friendly tarball of the paper sources |
| `python setup.py install`       | Install minimal requirements for GPy |
| `./asvrun.sh` | run performance benchmarks |

To build the paper, the packages `epstool` and `epstopdf` are required. Developers should also have `sphinx sphinx_rtd_theme matplotlib GPy codecov pylint parameterized pandas contexttimer` installed.

### Roadmap

0. LMC class refactor (to allow multi-input-dimension grids)
   -> elim ParameterValues -> FunctionalKernel
   -> LMCKernel -> LMCDerivative, Derivative -> ?
   -> GridKernel -> NonuniformKernel (subclass) Kernel = Matrix
   -> cache -> use lru_cache + `clear_cache()`
   -> prediction interface -- return full variance (cached?)
   -> get rid of sample prediction.
   -> reorg but keep Non-unif (time series) LMC -> InterpolatedLLGP
       -> runtime-only check for 1-d input
       -> Corresponding modifications in `README.md, benchmarks/benchlib/standard_tester.py, benchmarks/picture-fx2007/fxmetrics.py, benchmarks/picture-fx2007/fxpics.py, examples/example.ipynb, examples/fx2007.ipynb, runlmc/models/test_lmc.py`
   -> make new LLGP class, which assumes uniformity
   -> Separate out non-model functionality:
       -> prediction
       -> parallelism
       -> exact kernel version needs to be cleaned
        (current is hack).
       -> find chunks during initialization to pull out
       -> SLFM/indep GP story needs to be solid
   -> (?) test `LMC._raw_predict` unit testing, by using K_SKI() and anlogous math
0. Add basic components for block-toeplitz
0. Add [dataset](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/11998/12177)
0. Automatically find `min_grad_ratio` parameter. 
    * validate on small subset to get min grad ratio?
    * use quadratic form as a proxy?
    * Logdet approximations: (1) [Chebyshev-Hutchinson](https://arxiv.org/abs/1503.06394) [Code](https://sites.google.com/site/mijirim/logdet) (2) [Integral Probing](https://arxiv.org/abs/1504.02661) (3) [Lanczos](http://www-users.cs.umn.edu/~saad/PDF/ys-2016-04.pdf) (4) approx `tr log (A)` with MVM from [f(A)b](http://epubs.siam.org/doi/abs/10.1137/090778250) paper
0. Preconditioning
    * Cache Krylov solutions over iterations?
    * Cutajar 2016 iterative inversion approach?
    * T.Chan preconditioning for specialized on-grid case (needs development of partial grid)
0. TODO(test) - document everything that's missing documentation along the way.
0. Compare to [MTGP](http://www.robots.ox.ac.uk/~davidc/publications_MTGP.php), [CGP](http://www.jmlr.org/papers/volume12/alvarez11a/alvarez11a.pdf)
0. Minor perf improvements: what helps?
    * CPython; numba.
    * In-place multiplication where possible
    * square matrix optimizations
    * TODO(sparse-derivatives)
0. TODO(sum-fast) low-rank dense multiplications give SumKernel speedups?
0. multidimensional inputs and ARD.
0. TODO(prior). Compare to [spike and slab](http://www.aueb.gr/users/mtitsias/publications.html), also try MedGP (e.g., three-parameter beta) - add tests for priored versions of classes, some tests in parameterization/ (priors should be value-cached, try to use an external package)
0. HalfLaplace should be a Prior, add vectorized priors (remembering the shape)
0. Migrate to asv, separate tests/ folder (then no autodoc hack to skip test_* modules; pure-python benchmarks enable validation of weather/ and fx2007 benchmarks on travis-ci but then need to be decoupled from MATLAB implementations)
0. mean functions
0. product kernels (multiple factors) 
0. active dimension optimization

### Considerations 

* Real datasets: 
* Consider other approximate inverse algorithms: see Thm 2.4 of [Agarwal, Allen-Zhu, Bullins, Hazan, Ma 2016](https://arxiv.org/abs/1611.01146)

