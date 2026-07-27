#!/bin/bash
set -euo pipefail

# Launch sbatch jobs in dependent (chained) mode so subsequent runs resume from checkpoints.
# Default parameters can be overridden via environment variables.

ACCOUNT="${ACCOUNT:-huytran1-ic}"
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
    local sbatch_args=()
    [ -n "${ACCOUNT:-}" ] && sbatch_args+=(--account="$ACCOUNT")
    [ -n "${PARTITION:-}" ] && sbatch_args+=(--partition="$PARTITION")
    [ -n "${TIME_LIMIT:-}" ] && sbatch_args+=(--time="$TIME_LIMIT")
    [ -n "$dep_id" ] && sbatch_args+=(--dependency=afterany:"$dep_id")

    local out
    out=$(sbatch "${sbatch_args[@]}" "$SCRIPT" 2>&1) || {
        echo "sbatch submission failed: $out" >&2
        exit 1
    }
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
