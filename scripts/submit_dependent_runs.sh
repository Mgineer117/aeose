#!/bin/bash

# This script submits a specified sbatch script 22 times, chaining them
# so that each subsequent run waits for the previous one to finish.
# This is useful for long-running jobs that need to resume from checkpoints.

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <sbatch_script.sbatch> [additional_sbatch_args...]"
    exit 1
fi

SCRIPT=$1
shift

PARTITION="csl"
ACCOUNT="huytran-ic"
TIME_LIMIT="7-00:00:00"
NUM_RUNS=22

echo "Starting submission of $NUM_RUNS dependent jobs for script: $SCRIPT"

# Submit the first job
echo "Submitting run 1 of $NUM_RUNS..."
JOB_ID=$(sbatch --parsable --partition="$PARTITION" --account="$ACCOUNT" --time="$TIME_LIMIT" "$@" "$SCRIPT")
echo "Job 1 submitted with ID: $JOB_ID"

# Submit the remaining dependent jobs
for i in $(seq 2 $NUM_RUNS); do
    echo "Submitting run $i of $NUM_RUNS (dependent on $JOB_ID)..."
    # Using afterany so the next job runs even if the previous one reaches its 24h limit
    JOB_ID=$(sbatch --parsable --partition="$PARTITION" --account="$ACCOUNT" --time="$TIME_LIMIT" --dependency=afterany:$JOB_ID "$@" "$SCRIPT")
    echo "Job $i submitted with ID: $JOB_ID"
done

echo "Successfully submitted all $NUM_RUNS dependent jobs."
