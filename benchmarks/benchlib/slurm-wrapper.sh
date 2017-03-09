#!/bin/bash
# Internal slurm wrapper when it's missing.
# TODO needs documentation.

if which sbatch >/dev/null 2>/dev/null; then
    jobid=$(sbatch --parsable "$@")
    echo $jobid
    srun --dependency=afterany:$jobid true
else
    echo 'sbatch not found; running slurm wrapper on local machine'
    script="$1"
    shift
    array=$(grep "#SBATCH --array=" $script | head -1)
    array=$(echo $array | cut -c17- | tr '-' ' ')
    for i in $(seq $array); do
        echo "Running array job $i"
        outfile="slurm-out-$i.txt"
        errfile="slurm-err-$i.txt"
        SLURM_ARRAY_TASK_ID=$i $script "$@" >$outfile 2>$errfile
    done
fi

    
        

