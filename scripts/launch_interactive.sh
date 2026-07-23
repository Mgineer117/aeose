#!/bin/bash
# ---------------------------------------------------------------------------
# Interactively launch aeos runs, either on the local machine or on SLURM.
#
#   local   : starts one `python3 main.py` per (arch, seed) in the background
#             on a GPU you pick — the same thing a run_aeose_*.sbatch body does,
#             minus the scheduler.
#   cluster : generates an sbatch script from your answers and submits it,
#             optionally as a checkpoint-restart chain via launch_chain.sh
#             (the same mechanism launch_all.sh uses).
#
# For cluster mode the partition prompt lists only partitions that currently
# have idle/mixed nodes with unallocated GPUs, newest `sinfo` data, so you are
# not queueing behind a full partition by accident.
#
# Every prompt accepts a blank answer to take the shown default.
#
# Usage (run from anywhere):
#   bash scripts/launch_interactive.sh
#
# Environment overrides skip the matching prompt (handy for re-runs):
#   MODE=local ENV_NAME=charge GPU_IDX=1 bash scripts/launch_interactive.sh
#   MODE=cluster ENV_NAME=desat PARTITION=gpuA100x8 bash scripts/launch_interactive.sh
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
# Launch/submit from the repo root so the relative `python3 main.py` resolves
# (SLURM runs a job from the directory it was submitted from).
cd "$REPO_ROOT"

AVAILABLE_ENVS=(charge desat downlink resource)

# ask VAR "prompt" "default" -- sets VAR unless it is already non-empty.
ask() {
    local __var=$1 __prompt=$2 __default=$3 __reply
    if [ -n "${!__var:-}" ]; then
        echo "$__prompt ${!__var}  (from environment)"
        return
    fi
    read -r -p "$__prompt [$__default] " __reply
    printf -v "$__var" '%s' "${__reply:-$__default}"
}

echo "=== aeos interactive launcher ($REPO_ROOT) ==="
echo

# --- 1. where to run ------------------------------------------------------
if [ -z "${MODE:-}" ] && ! command -v sbatch >/dev/null 2>&1; then
    MODE=local
    echo "No sbatch on PATH -> mode: local"
else
    ask MODE "Run on 'local' machine or 'cluster' (SLURM)?" "cluster"
fi
case "$MODE" in
    local|cluster) ;;
    *) echo "error: mode must be 'local' or 'cluster', got '$MODE'" >&2; exit 1 ;;
esac

# --- 2. environment -------------------------------------------------------
echo
echo "Environments: ${AVAILABLE_ENVS[*]}"
ask ENV_NAME "Which env?" "downlink"
if ! printf '%s\n' "${AVAILABLE_ENVS[@]}" | grep -qx "$ENV_NAME"; then
    echo "error: unknown env '$ENV_NAME' (expected one of: ${AVAILABLE_ENVS[*]})" >&2
    exit 1
fi

# --- 3. architectures and seeds ------------------------------------------
# Semicolon-separated list; each entry is passed verbatim to --actor-fc-dim,
# so "64 64" means a two-layer MLP.
echo
echo "Actor architectures: semicolon-separated, spaces = extra layers."
echo "  e.g.  1;4;16        (small)      or   64 64;256 256;1024 1024   (large)"
ask ARCHS "Which architectures?" "1;4;16"
ask SEEDS "Which seeds (space-separated)?" "1 2 3"
ask NUM_WORKERS "Workers per run?" "3"
ask PROJECT "WandB project?" "aeos"

IFS=';' read -r -a ARCH_LIST <<< "$ARCHS"
read -r -a SEED_LIST <<< "$SEEDS"
TOTAL=$(( ${#ARCH_LIST[@]} * ${#SEED_LIST[@]} ))
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# arch_tag "64 64" -> 64x64   (filename-safe architecture label)
arch_tag() { local IFS=x; echo "$*"; }

# ===========================================================================
# CLUSTER MODE
# ===========================================================================
if [ "$MODE" = cluster ]; then
    # --- partitions with free GPUs ---------------------------------------
    # Sum (Gres - GresUsed) GPUs over nodes in idle/mixed state, per partition.
    # Gres looks like "gpu:a100:8(S:0-7)", GresUsed like "gpu:a100:3(IDX:0-2)".
    echo
    echo "Scanning partitions for idle/mixed nodes with free GPUs..."
    FREE_TABLE=$(sinfo -h -N -O 'Partition:40,StateLong:20,Gres:60,GresUsed:60' 2>/dev/null \
      | awk '
        function gpus(s,   n) {
            # first "gpu:...:<count>" field, ignoring any trailing (...) detail
            if (match(s, /gpu:[^ ]*/)) {
                n = substr(s, RSTART, RLENGTH)
                sub(/\(.*/, "", n)
                sub(/.*:/, "", n)
                if (n ~ /^[0-9]+$/) return n
            }
            return 0
        }
        {
            state = $2
            if (state != "idle" && state != "mixed") next
            part = $1; sub(/\*$/, "", part)     # strip default-partition marker
            free = gpus($3) - gpus($4)
            if (free > 0) { total[part] += free; nodes[part] += 1 }
        }
        END { for (p in total) printf "%s %d %d\n", p, total[p], nodes[p] }
      ' | sort -k2 -nr) || true

    if [ -n "$FREE_TABLE" ]; then
        echo
        printf "  %-3s %-24s %-10s %s\n" "#" "PARTITION" "FREE GPUS" "IDLE/MIX NODES"
        PART_NAMES=()
        while read -r p free nodes; do
            PART_NAMES+=("$p")
            printf "  %-3s %-24s %-10s %s\n" "${#PART_NAMES[@]}" "$p" "$free" "$nodes"
        done <<< "$FREE_TABLE"
        echo
        echo "Pick a number from the list, or type a partition name directly."
        ask PARTITION "Which partition?" "${PART_NAMES[0]}"
        # A bare number selects from the table above.
        if [[ "$PARTITION" =~ ^[0-9]+$ ]]; then
            if [ "$PARTITION" -lt 1 ] || [ "$PARTITION" -gt "${#PART_NAMES[@]}" ]; then
                echo "error: choice '$PARTITION' is out of range 1-${#PART_NAMES[@]}" >&2
                exit 1
            fi
            PARTITION="${PART_NAMES[$((PARTITION - 1))]}"
            echo "  -> $PARTITION"
        fi
    else
        echo "  (no idle/mixed nodes with free GPUs right now, or sinfo lacks"
        echo "   GresUsed support — falling back to a plain partition list)"
        sinfo -h -o '  %P  avail=%a  nodes=%D  state=%T' 2>/dev/null | sort -u || true
        ask PARTITION "Which partition?" "gpuA100x8"
    fi

    ask ACCOUNT "Which account?" "bhqw-delta-gpu"
    ask GPUS_PER_NODE "GPUs per node to request?" "2"
    ask CPUS_PER_TASK "CPUs per task?" "32"
    ask MEM "Memory?" "240G"
    ask WALLTIME "Wall time?" "2-00:00:00"
    ask NUM_CHUNKS "How many chained chunks (1 = no chaining)?" "3"

    if ! [[ "$GPUS_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
        echo "error: --gpus-per-node must be a positive integer, got '$GPUS_PER_NODE'" >&2
        exit 1
    fi

    GEN_DIR="$REPO_ROOT/scripts/generated"
    SBATCH_FILE="$GEN_DIR/aeose_${ENV_NAME}_${TIMESTAMP}.sbatch"

    echo
    echo "--- plan -------------------------------------------------------------"
    echo "  mode       : cluster (sbatch)"
    echo "  env        : $ENV_NAME"
    echo "  partition  : $PARTITION   account: $ACCOUNT"
    echo "  resources  : ${GPUS_PER_NODE} gpu(s), ${CPUS_PER_TASK} cpus, $MEM, $WALLTIME"
    echo "  archs      : ${ARCH_LIST[*]}"
    echo "  seeds      : ${SEED_LIST[*]}"
    echo "  runs       : $TOTAL  (spread round-robin over $GPUS_PER_NODE gpu(s))"
    echo "  workers    : $NUM_WORKERS  (=> $(( TOTAL * NUM_WORKERS )) worker processes)"
    echo "  chunks     : $NUM_CHUNKS"
    echo "  sbatch file: $SBATCH_FILE"
    echo "----------------------------------------------------------------------"
    read -r -p "Submit? [y/N] " CONFIRM
    case "${CONFIRM:-n}" in
        y|Y|yes|YES) ;;
        *) echo "Aborted."; exit 0 ;;
    esac

    mkdir -p "$GEN_DIR"
    {
        echo "#!/bin/bash"
        echo "#SBATCH --job-name=aeose_${ENV_NAME}"
        echo "#SBATCH --account=${ACCOUNT}"
        echo "#SBATCH --partition=${PARTITION}"
        echo "#SBATCH --nodes=1"
        echo "#SBATCH --gpus-per-node=${GPUS_PER_NODE}"
        echo "#SBATCH --cpus-per-task=${CPUS_PER_TASK}"
        echo "#SBATCH --mem=${MEM}"
        echo "#SBATCH --time=${WALLTIME}"
        echo "#SBATCH --output=aeose_${ENV_NAME}.o%j"
        echo "#SBATCH --mail-type=FAIL"
        echo "#SBATCH --mail-user=${MAIL_USER:-minjae5@illinois.edu}"
        echo
        echo "# Generated by scripts/launch_interactive.sh on $(date)"
        echo "source ~/.bashrc"
        echo "conda activate aeos"
        echo
        echo "export WANDB_INIT_TIMEOUT=300   # parallel launches init slowly"
        echo
        i=0
        for arch in "${ARCH_LIST[@]}"; do
            read -r -a arch_dims <<< "$arch"
            for seed in "${SEED_LIST[@]}"; do
                gpu=$(( i % GPUS_PER_NODE ))
                echo "python3 main.py --project $PROJECT --env-name $ENV_NAME --gpu-idx $gpu --num-workers $NUM_WORKERS --actor-fc-dim ${arch_dims[*]} --seed $seed &"
                i=$(( i + 1 ))
                # Stagger so concurrent WandB inits do not stampede.
                if [ $(( i % GPUS_PER_NODE )) -eq 0 ]; then echo "sleep 10"; fi
            done
        done
        echo
        echo "wait"
    } > "$SBATCH_FILE"
    chmod +x "$SBATCH_FILE"

    echo
    echo "Wrote $SBATCH_FILE"
    bash "$SCRIPT_DIR/launch_chain.sh" "$SBATCH_FILE" "$NUM_CHUNKS"
    echo
    echo "Inspect the queue with:  squeue -u \$USER"
    exit 0
fi

# ===========================================================================
# LOCAL MODE
# ===========================================================================
if command -v nvidia-smi >/dev/null 2>&1; then
    echo
    echo "Visible GPUs:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total \
               --format=csv,noheader || true
fi
ask GPU_IDX "Which GPU index?" "0"
if ! [[ "$GPU_IDX" =~ ^[0-9]+$ ]]; then
    echo "error: GPU index must be a non-negative integer, got '$GPU_IDX'" >&2
    exit 1
fi

LOG_DIR="$REPO_ROOT/log/interactive/${ENV_NAME}_gpu${GPU_IDX}_${TIMESTAMP}"

echo
echo "--- plan -------------------------------------------------------------"
echo "  mode       : local"
echo "  env        : $ENV_NAME"
echo "  gpu        : $GPU_IDX"
echo "  archs      : ${ARCH_LIST[*]}"
echo "  seeds      : ${SEED_LIST[*]}"
echo "  workers    : $NUM_WORKERS  (=> $(( TOTAL * NUM_WORKERS )) worker processes total)"
echo "  runs       : $TOTAL"
echo "  stdout logs: $LOG_DIR"
echo "----------------------------------------------------------------------"
read -r -p "Launch? [y/N] " CONFIRM
case "${CONFIRM:-n}" in
    y|Y|yes|YES) ;;
    *) echo "Aborted."; exit 0 ;;
esac

mkdir -p "$LOG_DIR"
export WANDB_INIT_TIMEOUT=${WANDB_INIT_TIMEOUT:-300}  # parallel launches init slowly

PIDS=()
for arch in "${ARCH_LIST[@]}"; do
    read -r -a arch_dims <<< "$arch"
    tag=$(arch_tag "${arch_dims[@]}")
    for seed in "${SEED_LIST[@]}"; do
        log_file="$LOG_DIR/${ENV_NAME}_fc${tag}_seed${seed}.log"
        python3 main.py \
            --project "$PROJECT" \
            --env-name "$ENV_NAME" \
            --gpu-idx "$GPU_IDX" \
            --num-workers "$NUM_WORKERS" \
            --actor-fc-dim "${arch_dims[@]}" \
            --seed "$seed" \
            > "$log_file" 2>&1 &
        PIDS+=($!)
        echo "launched pid $! : fc=[${arch_dims[*]}] seed=$seed -> $log_file"
    done
    # Stagger architecture groups so WandB init does not thundering-herd.
    sleep 10
done

echo
echo "$TOTAL run(s) launched. Tail them with:"
echo "  tail -f $LOG_DIR/*.log"
echo "Waiting for all runs (Ctrl-C detaches this shell; runs keep going)..."
wait "${PIDS[@]}"
echo "All runs finished."
