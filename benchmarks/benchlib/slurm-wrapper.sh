#!/bin/bash
# Internal slurm wrapper when it's missing.
# TODO needs documentation.

if which sbatch; then
    sbatch "$@"
else
    echo 'sbatch not found; running slurm wrapper on local machine'
    script="$1"
    shift
    array=$(grep "#SBATCH --array=" $script | head -1)
    array=$(echo $array | cut -c17- | tr '-' ' ')
    for i in $(seq $array); do
        echo "Running array job $i"
        outfile="slurm-wrapper-out-$i.txt"
        errfile="slurm-wrapper-err-$i.txt"
        SLURM_ARRAY_TASK_ID=$i $script "$@" >$outfile 2>$errfile
    done
fi

    
        

