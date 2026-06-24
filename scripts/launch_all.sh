#!/bin/bash
# ---------------------------------------------------------------------------
# Submit a chained (checkpoint-restart) experiment for EVERY environment.
#
# For each env sbatch script it calls launch_chain.sh, which submits
# NUM_CHUNKS jobs linked by `afterany` dependencies — each chunk resumes the
# previous chunk's checkpoint, so a run survives the cluster time limit.
#
# This launcher only needs `sbatch` on PATH (the SLURM client); it does NOT
# need conda or python. The training itself runs inside each .sbatch on the
# compute node, which activates conda there (`conda activate aeos`).
#
# Usage (run from anywhere):
#   bash scripts/launch_all.sh            # 3 chunks per env (default)
#   bash scripts/launch_all.sh 4          # 4 chunks per env
#   ENVS="charge downlink" bash scripts/launch_all.sh   # subset of envs
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
# Submit from the repo root so the relative `python3 main.py` in each sbatch
# resolves on the compute node (SLURM runs the job from the submit dir).
cd "$REPO_ROOT"

NUM_CHUNKS=${1:-${NUM_CHUNKS:-3}}
# Space-separated env list; defaults to all four.
read -r -a ENV_LIST <<< "${ENVS:-charge desat downlink resource}"

echo "Launching ${#ENV_LIST[@]} chained experiment(s), $NUM_CHUNKS chunk(s) each, from $REPO_ROOT"
echo

for env in "${ENV_LIST[@]}"; do
    sbatch_script="scripts/run_aeose_${env}.sbatch"
    if [ ! -f "$sbatch_script" ]; then
        echo "Skipping '$env': $sbatch_script not found" >&2
        continue
    fi
    echo "=== $env ==="
    bash "$SCRIPT_DIR/launch_chain.sh" "$sbatch_script" "$NUM_CHUNKS" \
        || echo "warning: failed to submit chain for '$env'; continuing with the rest" >&2
    echo
done

echo "All chains submitted. Inspect the queue with:  squeue -u \$USER"
