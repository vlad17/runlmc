# Benchmarks

This directory contains benchmarking code, invocations, and sample results.

## All actual benchmarks (everything but lib)

Each directory contains a `run.sh` file. This must be invoked from the respective directory. Each benchmark directory also has a corresponding `out` folder as a subfolder. Exceptions include the benchmarks that are auto-run by `asv`, which are located in the `asv` directory. Their outputs are at the top-level (postfixed with `-out`) because of `asv` requirements.

General info:

* Each `run.sh` must **not** be sourced
* `run.sh --help` prints additional info and details about the benchmark
* `run.sh --validate` should run a small version of the benchmark and validate the configuration.
* Benchmarks some use SLURM if it is installed.

## benchlib

**Internal** benchmarking API invoked from more specific benchmarking scripts.

Contains all benchmarking functions for both LLGP and COGP (another multi-output GP model, which uses MATLAB).

This is not a Python package. Files here must be manually added to the path and called.


