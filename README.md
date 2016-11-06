# runlmc

Do you like to apply Bayesian nonparameteric methods to your regressions? Are you frequently tempted by the flexibility that kernel-based learning provides? Do you have trouble getting structured kernel interpolation or various training conditional inducing point approaches to work in a non-stationary multi-output setting?

If so, this package is for you.

**runlmc** is a python 3 package designed to extend structural efficiencies from _Scalable inference for structured Gaussian process models_ (Staaçi 2012) and _Thoughts on Massively Scalable Gaussian Processes_ (Wilson et al 2015) to the non-stationary setting of linearly coregionalized multiple-output regressions. For the single output setting, MATLAB implementations are available [here](http://www.gaussianprocess.org/gpml/code/matlab/doc/).

## A note on GPy

[GPy](https://github.com/SheffieldML/GPy) is a way more general GP library that was a strong influence in the development of this one. I've tried to stay as faithful as possible to its structure.

Assuming this project proves successful, I'll try to integrate with GPy.

## Example

Currently, I'm only supporting 1 input dimension.

    def noisify(x, sd): return x + np.rand.normal(0, sd)
    Xs = [np.arange(-10, 5, 0.5), np.arange(-5.25, 10, .5)]
    Xs = [[noisify(x, 0.1) for x in ls] for ls in Xs]
    Ks = [runlmc.kern.StdPeriodic(...) for i in range(2)]
    Ys = runlmc.Sample(Xs, Ks, added_noise=[1,0.3])
    K = runlmc.LMC(input_dim=1,num_outputs=2,ranks=5,kernels_list=Ks)
    model = runlmc.GP(Xs, Ys, K)
    m.optimize()
    
    xticks = np.arange(-10, 10, 0.01)
    mus = m.mean(xticks)
    los, his = m.ci(xticks, 0.95)
    
    for i in range(2)
        print("Output", i)
        for outs in [lo, mu, hi]: plt.plot(xticks, outs[i])
        plt.plot(Xs[i], Ys[i])
        plt.show()

## Dev requirements

`pylint nose`


## Dev roadmap:

1. Toeplitz matrix representation, efficient eigendecomposition, and multiplication
2. Same as above, but for Kronecker matrices
3. Linear conjugate-gradient descent for fast inversion
4. Writing out the top-level GP inference and learning code (translating the math equations in the introduction to a usable API) -> make this work for IMC first (should be identical to SKI), then LMC!
5. Incorporating priors (e.g., three-parameter beta)
6. Log-determinant bound (greedy algorithm; may be improved with Prof. Tarjan)
7. Sphinxdoc + Doctests
8. Continuous integration for unit tests + pylint

