#!/bin/bash

set -e

rm -rf doc/_generated
sphinx-apidoc --separate --force --output-dir=doc/_generated runlmc/ 
mkdir -p doc/_generated/_static
PYTHONPATH=. sphinx-build -E -j $(nproc) -c doc/ -b html doc/_generated doc/_generated/_build/

