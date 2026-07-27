#!/bin/bash
set -euo pipefail

# Launch sbatch jobs in dependent (chained) mode so subsequent runs resume from checkpoints.
# Default parameters can be overridden via environment variables.

ACCOUNT="${ACCOUNT:-huytran-ic}"
PARTITION="${PARTITION:-csl}"
TIME_LIMIT="${TIME_LIMIT:-7-00:00:00}"

if [ -z "${1:-}" ]; then
    echo "Usage: ./launch_chain.sh <path_to_sbatch_script> [num_chunks]"
    echo "Example: ./launch_chain.sh scripts/run_aeose_charge.sbatch 3"
    exit 1
fi

SCRIPT=$1
NUM_CHUNKS="${2:-3}"

if [ ! -f "$SCRIPT" ]; then
    echo "Error: sbatch script '$SCRIPT' not found."
    exit 1
fi

# Submit one job, chained after previous job ID with afterany
submit() {
    local dep_id=$1
    local out
    if [ -z "$dep_id" ]; then
        out=$(sbatch --account="$ACCOUNT" --partition="$PARTITION" --time="$TIME_LIMIT" "$SCRIPT")
    else
        out=$(sbatch --account="$ACCOUNT" --partition="$PARTITION" --time="$TIME_LIMIT" --dependency=afterany:"$dep_id" "$SCRIPT")
    fi
    local job_id
    job_id=$(echo "$out" | awk '{print $4}')
    if ! [[ "$job_id" =~ ^[0-9]+$ ]]; then
        echo "Error: failed to parse job id from sbatch output: '$out'" >&2
        exit 1
    fi
    echo "$job_id"
}

echo "Chaining $NUM_CHUNKS dependent runs of $SCRIPT (Account: $ACCOUNT, Partition: $PARTITION, Time: $TIME_LIMIT)..."

PREV=""
for ((i = 1; i <= NUM_CHUNKS; i++)); do
    JOB=$(submit "$PREV")
    if [ -z "$PREV" ]; then
        echo "Submitted chunk $i (Job ID: $JOB)"
    else
        echo "Submitted chunk $i (Job ID: $JOB, dependent on $PREV)"
    fi
    PREV=$JOB
done

echo "All $NUM_CHUNKS dependent chunks queued successfully!"
