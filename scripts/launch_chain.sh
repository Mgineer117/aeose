#!/bin/bash

# Ensure a script is provided
if [ -z "$1" ]; then
    echo "Usage: ./launch_chain.sh <path_to_sbatch_script>"
    echo "Example: ./launch_chain.sh run_aeose_desat.sbatch"
    exit 1
fi

SCRIPT=$1

echo "Chaining 3 runs of $SCRIPT..."

# Submit the first 48h chunk
JOB1=$(sbatch $SCRIPT | awk '{print $4}')
echo "Submitted chunk 1 (Job ID: $JOB1)"

# Submit the second 48h chunk (waits for JOB1 to finish or get killed)
JOB2=$(sbatch --dependency=afterany:$JOB1 $SCRIPT | awk '{print $4}')
echo "Submitted chunk 2 (Job ID: $JOB2)"

# Submit the final 48h chunk (waits for JOB2)
JOB3=$(sbatch --dependency=afterany:$JOB2 $SCRIPT | awk '{print $4}')
echo "Submitted chunk 3 (Job ID: $JOB3)"

echo "All 3 chunks are queued successfully!"
