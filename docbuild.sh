#!/bin/bash

set -e

sphinx-apidoc --separate --force --output-dir=doc/_generated runlmc/ 
PYTHONPATH=. sphinx-build -j $(nproc) -c doc/ -b html doc/_generated doc/_generated/_build/
