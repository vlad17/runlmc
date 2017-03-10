#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --job-name=llgp-rep-cmp
#SBATCH --mem=5G
#SBATCH --array=1-5
#SBATCH --output=slurm-out-%a.txt
#SBATCH --error=slurm-err-%a.txt

REPOROOT="$1"
OUTFOLDER="$2"
IS_VALIDATION="$3"

if $IS_VALIDATION ; then
    MATRIX_SIZE="100"
else
    MATRIX_SIZE="5000"
fi
    
cd $REPOROOT

#benchmarks/benchlib/inv-run.sh $MATRIX_SIZE # Warmup
benchmarks/benchlib/inv-run.sh $MATRIX_SIZE > $OUTFOLDER/inv-run-$SLURM_ARRAY_TASK_ID.txt

