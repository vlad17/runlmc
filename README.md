# runlmc

Do you like to apply Bayesian nonparameteric methods to your regressions? Are you frequently tempted by the flexibility that kernel-based learning provides? Do you have trouble getting structured kernel interpolation or various training conditional inducing point approaches to work in a non-stationary multi-output setting?

If so, this package is for you.

**runlmc** is a python 3 package designed to extend structural efficiencies from _Scalable inference for structured Gaussian process models_ (Staaçi 2012) and _Thoughts on Massively Scalable Gaussian Processes_ (Wilson et al 2015) to the non-stationary setting of linearly coregionalized multiple-output regressions. For the single output setting, MATLAB implementations are available [here](http://www.gaussianprocess.org/gpml/code/matlab/doc/).

## Example



## Dev roadmap:

1. Toeplitz matrix representation, efficient eigendecomposition, and multiplication
2. Same as above, but for Kronecker matrices
3. Linear conjugate-gradient descent for fast inversion
4. Writing out the top-level GP inference and learning code (translating the math equations in the introduction to a usable API) -> make this work for IMC first (should be identical to SKI), then LMC!
5. Incorporating priors (e.g., three-parameter beta)
6. Log-determinant bound (greedy algorithm; may be improved with Prof. Tarjan)
7. Sphinxdoc + Doctests
