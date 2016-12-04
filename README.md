# runlmc

Do you like to apply Bayesian nonparameteric methods to your regressions? Are you frequently tempted by the flexibility that kernel-based learning provides? Do you have trouble getting structured kernel interpolation or various training conditional inducing point approaches to work in a non-stationary multi-output setting?

If so, this package is for you.

**runlmc** is a Python 3 package designed to extend structural efficiencies from _Scalable inference for structured Gaussian process models_ (Staaçi 2012) and _Thoughts on Massively Scalable Gaussian Processes_ (Wilson et al 2015) to the non-stationary setting of linearly coregionalized multiple-output regressions. For the single output setting, MATLAB implementations are available [here](http://www.gaussianprocess.org/gpml/code/matlab/doc/).

In other words, this provides a matrix-free implementation of multi-output GPs for certain covariances. As far as I know, this is also the only matrix-free implementation for single-output GPs in python.

## A note on GPy

[GPy](https://github.com/SheffieldML/GPy) is a way more general GP library that was a strong influence in the development of this one. I've tried to stay as faithful as possible to its structure.

I've re-used a lot of the GPy code. The main issue with simply adding my methods to GPy is that the API used to interact between GPy's `kern`, `likelihood`, and `inference` packages centers around the `dL_dK` object, a matrix derivative of the likelihood with respect to covariance. The materialization of this matrix is the very thing my algorithm tries to avoid for performance.

If there is some quantifiable success with this approach then integration with GPy would be a reasonable next-step.

## Example

Currently, I'm only supporting 1 input dimension.

    def noisify(x, sd): return x + np.rand.normal(0, sd)
    Xs = [np.arange(-10, 5, 0.5), np.arange(-5.25, 10, .5)]
    Xs = [[noisify(x, 0.1) for x in ls] for i, ls in enumerate(Xs, 1)]
    Ks = [runlmc.kern.StdPeriodic() for _ in range(2)]
    Ys = runlmc.util.Sample(Xs, Ks, added_noise=[1,0.3])
    m = runlmc.models.LMC(Xs, Ys,ranks=[1,1],kernels_list=Ks)
    m.optimize()
    
    xticks = np.arange(-10, 10, 0.01)
    mus, _ = m.predict([xticks, xticks])
    los, his = m.predict_quantiles([xticks, xticks])
    
    %matplotlib inline
    import matplotlib.pyplot as plt
    for i in range(2)
        print("Output", i)
        for outs in [lo, mu, hi]: plt.plot(xticks, outs[i])
        plt.plot(Xs[i], Ys[i])
        plt.show()
        
## Benchmarks

Make sure that the directory root is in the `PYTHONPATH` when running the benchmarks. E.g., from the directory root:

    PYTHONPATH=. python benchmark/inversion.py
        
## Dev Stuff

Required packages for development (Python 3 versions): `pylint nose paramz gpy sphinx contexttimer`.

Note:

 * All below invocations should be done from the repo root.
 
 * All invocations rely on `bash` scripts.
 
| Command           | Purpose  |
| ----------------- | -------- |
| `./style.sh`      | Check style with pylint, ignoring TODOs and locally-disabled warnings. |
| `./docbuild.sh`   | Regenerate docs (index will be in `doc/_generated/_build/runlmc.html`) |
| `nosetests`       | Run unit tests |

### Roadmap

0. `test_utils.py` - add tests for this, document, rename to `testing_utils.py`
0. matrix .to_numpy() functionality (modify benchmarks and tests to use it, too)
0. try to figure out if SLFM approach will work (pg. 16 in vector-valued-lmc.pdf). Otherwise, use determinant bound and easy derivative of bound.
0. Log determinant bound grad
0. `__init__` docs should go in class doc
0. python checks for PSD in toeplitz / numpy when in debug?
0. Writing out the top-level GP inference and learning code (translating the math equations in the introduction to a usable API) -> make this work for IMC first (should be identical to SKI), then LMC!
0. Continuous integration for unit tests
0. Drop gpy dep (in non-tests)
0. multidimensional inputs and ARD.
0. Incorporating priors (e.g., three-parameter beta) - add tests for priored versions of classes, some tests in parameterization/ (priors should be value-cached, try to use an external package)

### Considerations

* MINRES or CG?
* How can we add good preconditioners? How much do they help?
* What are condition numbers in practice?
* Why are sparse eigensolvers poor?
