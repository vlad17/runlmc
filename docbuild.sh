#!/bin/bash

set -e

sphinx-apidoc -H runlmc -A "Vladimir Feinberg" --separate --force --output-dir=doc/_generated runlmc/
cp doc/index.rst doc/_generated/
PYTHONPATH=. sphinx-build -j $(nproc) -c doc/ -b html doc/_generated doc/_generated/_build/
