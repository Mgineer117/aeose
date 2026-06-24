#!/bin/bash
set -euo pipefail

# Ensure a script is provided
if [ -z "${1:-}" ]; then
    echo "Usage: ./launch_chain.sh <path_to_sbatch_script> [num_chunks]"
    echo "Example: ./launch_chain.sh run_aeose_desat.sbatch 3"
    exit 1
fi

SCRIPT=$1
NUM_CHUNKS=${2:-3}

if [ ! -f "$SCRIPT" ]; then
    echo "Error: sbatch script '$SCRIPT' not found."
    exit 1
fi

# Submit one job, optionally chained after $1 with afterany (so the next chunk
# runs even when the previous one is killed by the wall-clock limit). Echoes the
# new job id; aborts the chain if submission fails.
submit() {
    local dep_id=$1
    local out
    if [ -z "$dep_id" ]; then
        out=$(sbatch "$SCRIPT")
    else
        out=$(sbatch --dependency=afterany:"$dep_id" "$SCRIPT")
    fi
    local job_id
    job_id=$(echo "$out" | awk '{print $4}')
    if ! [[ "$job_id" =~ ^[0-9]+$ ]]; then
        echo "Error: failed to parse job id from sbatch output: '$out'" >&2
        exit 1
    fi
    echo "$job_id"
}

echo "Chaining $NUM_CHUNKS runs of $SCRIPT..."

PREV=""
for ((i = 1; i <= NUM_CHUNKS; i++)); do
    JOB=$(submit "$PREV")
    if [ -z "$PREV" ]; then
        echo "Submitted chunk $i (Job ID: $JOB)"
    else
        echo "Submitted chunk $i (Job ID: $JOB, after $PREV)"
    fi
    PREV=$JOB
done

echo "All $NUM_CHUNKS chunks are queued successfully!"
