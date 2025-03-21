#!/bin/bash
#SBATCH -J runtime                         # Job name
#SBATCH -o logs/runtime_%j.out                  # output file (%j expands to jobID)
#SBATCH -e logs/runtime_%j.err                  # error log file (%j expands to jobID)
#SBATCH -N 1                                 # Total number of nodes requested
#SBATCH -n 8                                 # Total number of cores requested
#SBATCH --get-user-env                       # retrieve the users login environment
#SBATCH --mem=100GB                           # server memory (MBs) requested (per node)
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
    for batch_number in {1..50}
    do
        CMD="python runtime.py -m $method -bn $batch_number -dp $DATA_PATH -op $OUTPUT_PATH"
        echo $CMD
        eval $CMD
    done
done