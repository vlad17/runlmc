# NOAA weather data

Generates fake data from a 2-input, 5-output SLFM kernel with Q=2 and RBF from various lengthscales.

Assumes that `runlmc` package and its benchmakr library `benchlib` are on the `PYTHONPATH`.

Creates numpy and MATLAB version of the input in the CWD. From this directory, try the following:

    PYTHONPATH:../..:../../benchmarks/benchlib python mkdata.py