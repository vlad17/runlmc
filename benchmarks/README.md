# Benchmarks

This directory contains benchmarking code, invocations, and sample results.

## All actual benchmarks (everything but lib)

Each directory contains a `run.sh` file. This must be invoked from the respective directrory.

General info:

* Each `run.sh` must **not** be sourced
* `run.sh --help` prints additional info and details about the benchmark
* `run.sh --validate` should run a small version of the benchmark and validate the configuration.
* Benchmarks some use SLURM if it is installed.

## benchlib

**Internal** benchmarking API invoked from more specific benchmarking scripts.

Contains all benchmarking functions for both LLGP and COGP (another multi-output GP model, which uses MATLAB).

This is not a Python package. Files here must be manually added to the path and called.


