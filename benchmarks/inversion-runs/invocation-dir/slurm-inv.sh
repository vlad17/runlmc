#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --job-name=inv-kernels
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vyf@princeton.edu
#SBATCH --mem=5G
#SBATCH --array=1-5

cd "$SLURM_SUBMIT_DIR/../../.."

# Warmup
benchmarks/inv-run.sh 5000
sleep 5
benchmarks/inv-run.sh 5000 > benchmarks/inversion-runs/invocation-dir/inv-run-$SLURM_ARRAY_TASK_ID.txt

