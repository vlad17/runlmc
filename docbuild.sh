#!/bin/bash

set -e

sphinx-apidoc -H runlmc -A "Vladimir Feinberg" --separate --force --output-dir=doc/_generated runlmc/ $(echo $(find . -iname "test_*.py"))
cp doc/index.rst doc/_generated/
cd doc
PYTHONPATH=.. sphinx-build -j $(nproc) -c . -b html _generated/ _generated/_build/
