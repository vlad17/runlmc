#!/bin/bash
# Runs fx2007 dataset example

USAGE="Usage: ./run.sh [--help|--validate]"
EXPECTED_DIR="runlmc/benchmarks/fx2007"
HELP_STR="$USAGE

Must be run from $EXPECTED_DIR.

Runs only on local machine, but leverages all processessors avaiable in the 
model evaluation phase.

This runs the fx2007 dataset for the SLFM model several times on several
different interpolation point counts. Each setting has its timing, SMSE,
and NLPD statistics outputted (mean and standard error).

Then, we run COGP on the same dataset.

Results are printed to a trace file, ./out/stdout-fx2007.txt

Flags
    --validate Run a small case to verify configuration.
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
        "--validate")
            IS_VALIDATION="true"
            ;;
        *)
            echo $USAGE >/dev/stderr
            exit 3
            ;;
    esac
else
    IS_VALIDATION="false"
fi

cd out/

OUTFOLDER=$PWD
REPOROOT=$(readlink -f "$PWD/../../../")

cd $REPOROOT

OMP_NUM_THREADS=1 PYTHONPATH="$REPOROOT:$REPOROOT/benchmarks/benchlib" python3 -u $REPOROOT/benchmarks/fx2007/fx2007.py $IS_VALIDATION 2>&1 | tee $OUTFOLDER/stdout-fx2007.txt | egrep -e '--->'
