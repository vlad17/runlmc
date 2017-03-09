#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --job-name=llgp-grad-grid
#SBATCH --mem=5G
#SBATCH --array=0-59
#SBATCH --output=slurm-out-%a.txt
#SBATCH --error=slurm-err-%a.txt

REPOROOT="$1"
OUTFOLDER="$2"
IS_VALIDATION="$3"

PYTHONPATH="$REPOROOT"

script="$REPOROOT/benchmarks/benchlib/bench.py"

QQs=(1 5 10)
EPSs=(1 0.1 0.01 0.001 0.0001)
KERNs=(rbf periodic matern mix)

# 3 * 5 * 4 = 60 total grid size
grid=($(echo {0..2}"+"{0..4}"+"{0..3}))

mine=${grid[$SLURM_ARRAY_TASK_ID]}

QQ=${QQs[$(echo $mine | cut -d"+" -f1)]}
EPS=${EPSs[$(echo $mine | cut -d"+" -f2)]}
KERN=${KERNs[$(echo $mine | cut -d"+" -f3)]}

if $IS_VALIDATION ; then
    if [[ "$Q" -eq 10 ]]; then
        exit 0
    fi
    if [[ "$EPS" != "1" && "$EPS" != "0.1" ]]; then
        exit 0
    fi
    SIZE="10"
    RANK="1"
    DIM="1"
else
    SIZE="500"
    RANK="3"
    DIM="10"

    # Warmup
    PYTHONPATH=$PYTHONPATH python3 $script $SIZE $DIM ${RANK} $QQ "$EPS" $KERN 1234 opt 
    PYTHONPATH=$PYTHONPATH python3 $script $SIZE $DIM ${RANK} $QQ "$EPS" $KERN 1234 opt 
fi

for i in $(seq 0 4); do
    PYTHONPATH=$PYTHONPATH python3 $script $SIZE $DIM ${RANK} $QQ "$EPS" $KERN "1234$i" opt > "$OUTFOLDER/n${SIZE}0-d${DIM}-r${RANK}-q$QQ-eps$EPS-k$KERN-run$i.txt"
done

