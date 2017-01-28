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

    lmc.optimize(optimizer=DerivFree()) # deriv-free for now
    # ... more plotting code
    
![opt](https://raw.githubusercontent.com/vlad17/runlmc/master/examples/opt.png)
        
### Running the Examples and Benchmarks

Make sure that the directory root is in the `PYTHONPATH` when running the benchmarks. E.g., from the directory root:

    PYTHONPATH=. python benchmark/inversion.py
    
Similarly, for examples:

    cd examples; PYTHONPATH=.. jupyter notebook example.ipynb
        
## Dev Stuff

Required packages for development (Python 3 versions): `pylint nose paramz gpy sphinx contexttimer numdifftools`.

All below invocations should be done from the repo root.
 
| Command           | Purpose  |
| ----------------- | -------- |
| `./style.sh`      | Check style with pylint, ignoring TODOs and locally-disabled warnings. |
| `./docbuild.sh`   | Regenerate docs (index will be in `doc/_generated/_build/runlmc.html`) |
| `./docpush.sh`   | Publish current docs (requries write access to repo) |
| `nosetests -l DEBUG`       | Run unit tests |

### Roadmap

0. Move to Cutajar approach. Use tests. Dedup/cleanup (benchmark code too).
0. Remove linalg approximate eigenvalue code, old benchmarks.
0. Re-run examples
0. Preconditioner?
    * Does chan Preconditioner carry over to SKI approximation?
    * Do other inner circulant preconditioners (e.g., whittle) help inversion?
0. Minor perf improvements: what helps? (MKL, CPython)
0. Apply to synthetic and real datasets [link1](http://www.robots.ox.ac.uk/~davidc/publications_MTGP.php) [spike and slab](http://www.aueb.gr/users/mtitsias/publications.html), also try MedGP.
0. Write up the current algorithm (PDF)
0. multidimensional inputs and ARD.
0. rank > 1

### Considerations 

* Investigate: when is iteration NOT converging (critical log) - what's the condition number in that case.
* SEED=3333617092 nosetests runlmc.models.test_lmc breaks Toeplitz PSD strictly (currently has large negative eigval cutoff)
* SLFM approach -> can we take determeinant in this representation?
   0. SLFM approach work for computing deriv of log det / log det exactly (pg. 16 in vector-valued-lmc.pdf)
   0. How to take determinant? Derivatives?
   0. Re-prove (legitimately); start by showing wilson SKI m^(-3) conv (in multioutput case), then prove SLFM for 1 input dim, rank 1
   0. Rank >1 reduction to rank 1 (use constant kq terms)
   0. multidimensional proof; requires cubic interpol from SKI (again, for multioutput)
   0. SLFM code up; GP code up; do K, dK/dL reconstruction experiments.
* TODO(MSGP) - fast toeplitz eig
* MINRES or LCG?
* How can we add good preconditioners? How much do they help?
* What are condition numbers in practice?
* Why are sparse eigensolvers poor? Can we use them as an accurate general-purpose solution if all else fails?
* Consider other approximate inverse algorithms: see Thm 2.4 of [Agarwal, Allen-Zhu, Bullins, Hazan, Ma 2016](https://arxiv.org/abs/1611.01146)
0. New logdet algo? [Chebyshev-Hutchinson](https://arxiv.org/abs/1503.06394) [Code](https://sites.google.com/site/mijirim/logdet)

### Low-priority Tasks

0. BSD 3-clause
0. Allow extrapolation in util.interpolation.py
0. document SKI
0. test multi_interpolant
0. test SKI
0. test lmc._autogrid for edge cases.
0. test `LMC._raw_predict` unit testing, by using K_SKI() and anlogous math
0. np.linalg.eigvalsh -> scipy.linalg.eigvalsh
0. rename `rand_psd` -> `rand_pd`
0. Continuous integration for unit tests
0. Drop gpy dep (in non-tests) - requires exact kernel cholesky impl
0. TODO(priors) - Incorporating priors (e.g., three-parameter beta) - add tests for priored versions of classes, some tests in parameterization/ (priors should be value-cached, try to use an external package)
0. product kernels (multiple factors) 
0. active dimension optimization

### Thesis Plan

0. Intro
0. Related Work - see detailed-plan.md; proposal; close loose ends here
0. SKI - prove O(m^-3) formalism (one-input case)
0. runlmc theoretical kernel re-creation error (can we re-prove? what are the bounds) (one-input case) (SLFM approach, if viable)
0. experimental proof of above
0. algorithm for runlmc kernel; explain/prove fast structural runtimes
0. experimental proof for above; comparison to gpy exact/ssgp
