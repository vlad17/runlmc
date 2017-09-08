#!/bin/bash
# runs asv benchmarks

PYTHONPATH="$PWD:$PWD/benchmarks/benchlib" asv run --python=python --verbose
