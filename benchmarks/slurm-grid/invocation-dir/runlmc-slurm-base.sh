#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --job-name=grid-kernels
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vyf@princeton.edu
#SBATCH --mem=5G
#SBATCH --array=0-59


REPOBASE="$SLURM_SUBMIT_DIR/../../.."

PYTHONPATH="$REPOBASE"

script="$REPOBASE/benchmarks/bench.py"

QQs=(1 5 10)
EPSs=(1 0.1 0.01 0.001 0.0001)
KERNs=(rbf periodic matern mix)

# 3 * 5 * 4 = 60 total grid size
grid=($(echo {0..2}"+"{0..4}"+"{0..3}))

mine=${grid[$SLURM_ARRAY_TASK_ID]}

QQ=${QQs[$(echo $mine | cut -d"+" -f1)]}
EPS=${EPSs[$(echo $mine | cut -d"+" -f2)]}
KERN=${KERNs[$(echo $mine | cut -d"+" -f3)]}
outfile=

# Warmup
PYTHONPATH=$PYTHONPATH python3 $script 500 10 3 $QQ "$EPS" $KERN 1234 opt 
PYTHONPATH=$PYTHONPATH python3 $script 500 10 3 $QQ "$EPS" $KERN 1234 opt 

for i in $(seq 0 4); do
    PYTHONPATH=$PYTHONPATH python3 $script 500 10 3 $QQ "$EPS" $KERN "1234$i" opt > "$REPOBASE/benchmarks/slurm-grid/invocation-dir/n5000-d10-r3-q$QQ-eps$EPS-k$KERN-run$i.txt"
done

