#!/bin/bash
#SBATCH -J acc                         # Job name
#SBATCH -o logs/acc_%j.out                  # output file (%j expands to jobID)
#SBATCH -e logs/acc_%j.err                  # error log file (%j expands to jobID)
#SBATCH -N 1                                 # Total number of nodes requested
#SBATCH -n 4                                 # Total number of cores requested
#SBATCH --get-user-env                       # retrieve the users login environment
#SBATCH --mem=32GB                           # server memory (MBs) requested (per node)
#SBATCH --gres=gpu:a6000:1

METHODS=(
    full
    performer
    reformer
    kdeformer
    scatterbrain
    thinformer
)
DATA_PATH=$1
OUTPUT_PATH=$2
for method in "${METHODS[@]}"
do
    for seed in 1 2 3 4 5
    do
        CMD="python accuracy.py -m1 $method -m2 $method -s $seed -dp $DATA_PATH -op $OUTPUT_PATH"
        echo $CMD
        eval $CMD
    done
done