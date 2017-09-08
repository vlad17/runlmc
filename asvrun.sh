#!/bin/bash
# runs asv benchmarks

PYTHONPATH="$PWD:$PWD/benchmarks/benchlib" asv run --verbose
asv publish
asv gh-pages

