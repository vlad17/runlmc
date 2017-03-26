#!/bin/bash
# Creates pictures from 1 run of fx2007 dataset example

USAGE="Usage: ./run.sh [--help]"
EXPECTED_DIR="runlmc/benchmarks/picture-fx2007"
HELP_STR="$USAGE

Must be run from $EXPECTED_DIR.

Runs only on local machine, but leverages all processessors avaiable in the 
model evaluation phase. Creates pretty pictures from a single run of
the fx2007 paper and puts them into ./out.

prediction_time.txt - trace of time it takes to do some predictions
fx2007graph.pdf - graphical comparison of predictive mean+var of LLGP vs COGP
                  on the stocks being predicted.
sample-pred-var.eps - exact vs sampled predictive variance estimation
iterations.eps - number of iterations MINRES requires during course of optimization
running_cutoff.eps - gradient norms (and cutoff) during course of optimization

stdout.txt - trace of execution stdout (this program)

Flags
    --help Print this help message.
"

base1=$(basename $PWD)
base2=$(cd .. && basename $PWD)
base3=$(cd .. && cd .. && basename $PWD)

if [[ "$base3/$base2/$base1" != "$EXPECTED_DIR" ]]; then
    echo "Must be run from $EXPECTED_DIR" >/dev/stderr
    exit 1
fi

if [[ "$base3/$base2/$base1" != "$EXPECTED_DIR" ]]; then
    echo "Must be run from $EXPECTED_DIR" >/dev/stderr
    exit 1
fi

if [[ $# -gt 1 ]]; then
    echo $USAGE >/dev/stderr
    exit 1
fi

if [[ $# -eq 1 ]]; then
    case $1 in
        "--help")
            printf "$HELP_STR"
            exit 0
            ;;
        *)
            echo $USAGE >/dev/stderr
            exit 3
            ;;
    esac
fi

cd out/

OUTFOLDER=$PWD
REPOROOT=$(readlink -f "$PWD/../../../")

cd $REPOROOT

OMP_NUM_THREADS=1 PYTHONPATH="$REPOROOT:$REPOROOT/benchmarks/benchlib" python3 -u $REPOROOT/../$EXPECTED_DIR/fxpics.py $OUTFOLDER | tee $OUTFOLDER/stdout.txt
# metrics should use all threads (capped at 1 proc)
PYTHONPATH="$REPOROOT:$REPOROOT/benchmarks/benchlib" python3 -u $REPOROOT/../$EXPECTED_DIR/fxmetrics.py $OUTFOLDER | tee -a $OUTFOLDER/stdout.txt
