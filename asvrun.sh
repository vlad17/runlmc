#!/bin/bash
# runs asv benchmarks, publishes them

set -e

currbranch=$(git rev-parse --abbrev-ref HEAD)

if ! git diff-index --quiet HEAD -- ; then
    echo 'uncommited changes present'
    exit 1
fi

PYTHONPATH="$PWD:$PWD/benchmarks/benchlib" asv run --verbose
asv publish
asv gh-pages || true

# asv gh-pages bug fix
# https://github.com/spacetelescope/asv/issues/515
git checkout gh-pages
git push -f
git checkout $(currbranch)



