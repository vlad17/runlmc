#!/bin/bash
# Runs representation comparison benchmarking script

USAGE="Usage: ./run.sh [--help|--validate]"
EXPECTED_DIR="runlmc/benchmarks/representation-cmp"
HELP_STR="$USAGE

Must be run from $EXPECTED_DIR.

Uses SLURM if it is available.

Runs a comparison between different representations of the grid
kernel on setups that are amenable to a variety of kernel shapes.
The comparison is made between the MINRES matrix inversion times,
which is dependent on the matrix-vector multiplication runtime.

This will produce output files in ./out directory.
This will overwrite any existing output files in that directory.
    results.txt - human-readable
    results.tex - LaTeX printout version

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
../../benchlib/slurm-wrapper.sh ../slurm-job.sh $REPOROOT $OUTFOLDER $IS_VALIDATION

echo
echo 'Gathering results'
echo

latex='
\\begin{tabular}{|ccc|cccc|}
  \\hline
  \\abovespace\\belowspace
  $D$ & $R$ & $Q$ & \\textsc{cholesky} & \\textsc{sum} & \\textsc{bt} & \\textsc{slfm}\\\\
\\hline
  \\abovespace
'
for i in 3 10 17; do
    
    title=$((i - 2))
    name=$(sed "${title}q;d" inv-run-1.txt)
    
    numbersonly=$(echo $name | sed -e 's/[^0-9]/ /g' -e 's/^ *//g' -e 's/ *$//g' | tr -s ' ' | sed 's/ /\n/g')

    tail -n +$i inv-run-1.txt | head -n 4 | tr -s " " | cut -d" " -f8 > "/tmp/cols"
    kerntype=$(sed "2q;d" inv-run-1.txt | cut -d" " -f12)

    echo $name | tee "results.txt"

    t=$(mktemp)

    for file in inv-run-*.txt; do
        tail -n +$i $file | head -n 4 | tr -s " " | cut -d" " -f4 > "${t}-column-$file"
    done

    paste $t-column-* > $t
    awk '{s=0; for (i=1;i<=NF;i++)s+=$i; print s/NF;}' $t > "/tmp/avgs"
    paste /tmp/avgs /tmp/cols | tee "results.txt"

    minnum=$(sort -n /tmp/avgs | head -n 1)

    # numbersonly contains n, d, r, q,
    # we only want d, r, q
    numbersonly=$(echo $numbersonly | cut -d' ' -f2-)    
    
    row=""
    for number in $numbersonly $(cat /tmp/avgs); do
        if [ "$number" = "$minnum" ]; then
            left="\\\\textbf{"
            right="}"
        else
            left=""
            right=""
        fi
        number=$(printf "%0.2f" $number)
        row="${row} & \$ ${left}${number}${right} \$"
    done
    row=$(echo $row | cut -c3-)
    newline=$'\n'
    latex="$latex $row \\\\\\\\ $newline"
done

echo
echo 'latex table in out/results.tex'

epilog='  \\belowspace \\\\

  \\hline
\\end{tabular}
'
printf "${latex::${#latex}-6}\n${epilog}" > results.tex
