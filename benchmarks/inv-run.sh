#!/bin/bash
# Usage: benchmarks/inv-run.sh [N] [kern] [eps]
# N is total GP size, default 100 (keep it divisible by 20)
# kern is kernel type, default mix
# eps is error mean, default 0.1
# see bench.py usage for more info

err=$(mktemp)
out=$(mktemp)

rm -f $out $err

if [ -z "$1" ]; then
    N="100"
else
    N="$1"
fi

if [ -z "$2" ]; then
    kern="mix"
else
    kern="$2"
fi

if [ -z "$3" ]; then
    eps="1e-1"
else
    eps="$3"
fi

echo "N=$N D=2 R=2 Q=10 SMALL DIM -> BT is best"
n=$((N / 2))
PYTHONPATH=. python3 benchmarks/bench.py $n 2 2 10 $eps $kern 1234 inv >$out 2>>$err
head -n1 $out
tail -n5 $out

echo "N=$N D=10 R=1 Q=10 SMALL RANK -> SLFM is best"
n=$((N / 10))
PYTHONPATH=. python3 benchmarks/bench.py $n 10 1 10 $eps $kern 1234 inv >$out 2>>$err
head -n1 $out
tail -n5 $out

echo "N=$N D=10 R=10 Q=1 SMALL SUM -> SUM"
n=$((N / 10))
PYTHONPATH=. python3 benchmarks/bench.py $n 10 10 1 $eps $kern 1234 inv >$out 2>>$err
head -n1 $out
tail -n5 $out

if [ -s $err ]; then
    echo "WARNINGS/ERRORS possible check $err"
fi
