#!/bin/bash
# Runs grid comparison for gradient benchmarking script

USAGE="Usage: ./run.sh [--help|--validate]"
EXPECTED_DIR="runlmc/benchmarks/grad-grid"
HELP_STR="$USAGE

Must be run from $EXPECTED_DIR.

Uses SLURM if it is available.

Runs a comparison between different representations of the grid
kernel on setups that are amenable to a variety of kernel shapes.
The comparison is made between the runtime and accuracy of data log
likelihood gradients.

This script invokes makepics.py.

This will produce output files in the ./out directory.
This will overwrite any existing output files in that directory.
    relalpha_l1.eps - L1 norm relative error from inversion
    relalpha_l2.eps - L2 norm relative error from inversion
    relgrad_l1.eps - L1 norm relative error from gradient
    relgrad_l2.eps - L2 norm relative error from gradient
    time_ratio.eps - ratio of time exact computation : llgp

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
../../benchlib/slurm-wrapper.sh ../slurm-job.sh $REPOROOT $OUTFOLDER $IS_VALIDATION

sumfile=extracted_summary.csv

rm -f $sumfile

QQs=(1 5 10)
EPSs=(1 0.1 0.01 0.001 0.0001)
KERNs=(rbf periodic matern mix)

if $IS_VALIDATION ; then
    SIZE="10"
else
    SIZE="500"
fi

echo "n,d,r,q,eps,k,time_ratio,relgrad_l1,relgrad_l2,relalpha_l1,relalpha_l2" > $sumfile

maxrun=4

for gridpoint in $(echo {0..2}"+"{0..4}"+"{0..3}) ; do
    mine=$gridpoint
    QQ=${QQs[$(echo $mine | cut -d"+" -f1)]}
    EPS=${EPSs[$(echo $mine | cut -d"+" -f2)]}
    KERN=${KERNs[$(echo $mine | cut -d"+" -f3)]}

    base="n${SIZE}0-d10-r3-q${QQ}-eps${EPS}-k${KERN}-run"

    nfile=$(find . -maxdepth 1 -name "$base*.txt" | wc -l)

    nrun=$(($maxrun + 1))
    if [ "$nfile" -ne "$nrun" ]; then
        echo "only $nfile of $nrun runs found for Q $QQ eps $EPS k $KERN, skipping"
        continue
    fi

    ratios1=""
    ratios2=""
    alphas1=""
    alphas2=""
    for i in $(seq 0 $maxrun); do
        file="$base$i.txt"
        grep -A 2 "total optimization iteration time" $file | tail -n 2 | tr -s " " | cut -d" " -f2 > /tmp/$file-times
        ratios1=$(echo $ratios1 $(grep "err:grad l1" $file | tr -s " " | cut -d" " -f2))
        ratios2=$(echo $ratios2 $(grep "err:grad l2" $file | tr -s " " | cut -d" " -f2))
        alphas1=$(echo $alphas1 $(grep "alpha l1" $file | tr -s " " | cut -d" " -f2))
        alphas2=$(echo $alphas2 $(grep "alpha l2" $file | tr -s " " | cut -d" " -f2))
    done

    avg_times=($(paste /tmp/$base*.txt-times | awk '{s=0; for (i=1;i<=NF;i++)s+=$i; print s/NF;}'))
    time_ratio=$(bc -l <<< "${avg_times[0]} / ${avg_times[1]}")
    
    ratios1a=(${ratios1})
    ratios2a=(${ratios2})
    alphas1a=(${alphas1})
    alphas2a=(${alphas2})

    if [ -z ${ratios1a[$maxrun]} ]; then
        echo error for ratios1 $ratios1 on "Q $QQ eps $EPS k $KERN, skipping"
        continue
    fi
    if [ -z ${ratios2a[$maxrun]} ]; then
        echo error for ratios2 $ratios2 on "Q $QQ eps $EPS k $KERN, skipping"
        continue
    fi
    if [ -z ${alphas1a[$maxrun]} ]; then
        echo error for alphas1 $alphas1 on "Q $QQ eps $EPS k $KERN, skipping"
        continue
    fi
    if [ -z ${alphas2a[$maxrun]} ]; then
        echo error for alphas2 $alphas2 on "Q $QQ eps $EPS k $KERN, skipping"
        continue
    fi

    function join_by { local d=$1; shift; echo -n "$1"; shift; printf "%s" "${@/#/$d}"; }

    r1add=$(join_by ' + ' $ratios1 | sed -e 's/[eE]+*/\*10\^/g')
    r2add=$(join_by ' + ' $ratios2 | sed -e 's/[eE]+*/\*10\^/g')
    a1add=$(join_by ' + ' $alphas1 | sed -e 's/[eE]+*/\*10\^/g')
    a2add=$(join_by ' + ' $alphas2 | sed -e 's/[eE]+*/\*10\^/g')

    avg_r1=$(bc -l <<< "( $r1add ) / ($maxrun + 1)")
    avg_r2=$(bc -l <<< "( $r2add ) / ($maxrun + 1)")
    avg_a1=$(bc -l <<< "( $a1add ) / ($maxrun + 1)")
    avg_a2=$(bc -l <<< "( $a2add ) / ($maxrun + 1)")
    
    echo "5000,10,3,${QQ},${EPS},${KERN},$time_ratio,$avg_r1,$avg_r2,$avg_a1,$avg_a2" >> $sumfile
done

echo >> $sumfile

python3 ../makepics.py
