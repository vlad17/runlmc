#!/bin/bash
# Runs representation comparison benchmarking script

EXPECTED_DIR="runlmc/benchmarks/representation-cmp"
HELP_STR='
Uses SLURM if it is available.

Runs a comparison between different representations of the grid
kernel on setups that are amenable to a variety of kernel shapes.
The comparison is made between the MINRES matrix inversion times,
which is dependent on the matrix-vector multiplicatoin runtime.
This will produce output files in ./out directory.
This will overwrite any existing output files in that directory.
out/results.txt will contain a human-readable and LaTeX printout.
'

base1=$(basename $PWD)
base2=$(cd .. && basename $PWD)
base3=$(cd .. && cd .. && basename $PWD)

if [[ "$base3/$base2/$base1" != "$EXPECTED_DIR" ]]; then
    echo "Must be run from $EXPECTED_DIR" >/dev/stderr
    exit 1
fi

../benchlib/run-skeleton.sh "$HELP_STR" "$@"
case $? in
    0)
        MATRIX_SIZE="5000"
        ;;
    1)
        MATRIX_SIZE="100"
        ;;
    2)
        echo "here"
        exit 0
        ;;
    3)
        exit 1
        ;;
esac

cd out/

OUTFOLDER=$PWD
REPOROOT=$(readlink -f "$PWD/../../../")
../../benchlib/slurm-wrapper.sh ../slurm-job.sh $REPOROOT $OUTFOLDER $MATRIX_SIZE

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
printf "${latex}\n${epilog}" > results.tex
