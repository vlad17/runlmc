#!/bin/bash

eps="1e-2"

echo 'N=4000 R=1 D=10 Q=10 SMALL RANK -> SLFM is best'
PYTHONPATH=. python3 benchmarks/bench.py 400 10 10 $eps mix 1234 inversion > /tmp/out
head -n1 /tmp/out
tail -n5 /tmp/out

echo 'N=4000 R=1 D=10 Q=1 SMALL SUM -> SUM is best' # move to high-rank
PYTHONPATH=. python3 benchmarks/bench.py 400 10 1 $eps mix 1234 inversion > /tmp/out
head -n1 /tmp/out
tail -n5 /tmp/out

echo 'N=4000 R=1 D=2 Q=1 SMALL DIM -> BT is best' # move to high-rank
PYTHONPATH=. python3 benchmarks/bench.py 2000 2 10 $eps mix 1234 inversion > /tmp/out
head -n1 /tmp/out
tail -n5 /tmp/out

echo 'N=4000 R=1 D=4 Q=4 GRADIENTS'
PYTHONPATH=. python3 benchmarks/bench.py 1000 4 4 $eps mix 1234 gradients > /tmp/out
head -n1 /tmp/out
grep -A 4 "matrix materialization" /tmp/out
tail -n5 /tmp/out

