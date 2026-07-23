#!/bin/bash
# ---------------------------------------------------------------------------
# Interactive front-end for the same sweep launch_all.sh submits.
#
# launch_all.sh submits scripts/run_aeose_<env>.sbatch for every env, each of
# which runs the fixed 18-run sweep (6 actor sizes x 3 seeds, split over 2
# GPUs). This script asks which envs, which model sizes, which seeds and which
# resources you want, then does the same thing with your answers:
#
#   cluster : generates an sbatch script per env and submits it as a
#             checkpoint-restart chain via launch_chain.sh (what launch_all.sh
#             does), after letting you pick a partition that has free GPUs.
#   local   : runs the same commands directly on this machine, round-robin
#             over the GPUs you name — no scheduler involved.
#
# Every prompt accepts a blank answer to take the shown default, so hitting
# <Enter> through the whole thing reproduces launch_all.sh exactly.
#
# Usage (run from anywhere):
#   bash scripts/launch_interactive.sh
#
# Environment overrides skip the matching prompt (handy for re-runs):
#   MODE=local ENVS=charge GPUS="0 1" bash scripts/launch_interactive.sh
#   MODE=cluster ENVS="charge desat" PARTITION=gpuA100x8 SIZE=large \
#       bash scripts/launch_interactive.sh
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
# Launch/submit from the repo root so the relative `python3 main.py` resolves
# (SLURM runs a job from the directory it was submitted from).
cd "$REPO_ROOT"

AVAILABLE_ENVS=(charge desat downlink resource)

# Model sizes, semicolon-separated; each entry goes verbatim to --actor-fc-dim,
# so "64 64" is a two-layer MLP. These match the run_aeose_*.sbatch sweep.
SIZES_SMALL="1;4;16"
SIZES_LARGE="64 64;256 256;1024 1024"
SIZES_FULL="$SIZES_SMALL;$SIZES_LARGE"

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

# arch_tag "64 64" -> 64x64   (filename-safe architecture label)
arch_tag() { local IFS=x; echo "$*"; }

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

# --- 2. environments (one or many, like ENVS= in launch_all.sh) -----------
echo
echo "Environments: ${AVAILABLE_ENVS[*]}   ('all' = every one, as launch_all.sh does)"
ask ENVS "Which env(s)?" "all"
if [ "$ENVS" = all ]; then
    ENV_LIST=("${AVAILABLE_ENVS[@]}")
else
    read -r -a ENV_LIST <<< "$ENVS"
fi
for env in "${ENV_LIST[@]}"; do
    if ! printf '%s\n' "${AVAILABLE_ENVS[@]}" | grep -qx "$env"; then
        echo "error: unknown env '$env' (expected: ${AVAILABLE_ENVS[*]} or 'all')" >&2
        exit 1
    fi
done

# --- 3. model sizes -------------------------------------------------------
echo
echo "Model sizes (--actor-fc-dim):"
echo "  1) full    $SIZES_FULL   <- the run_aeose_*.sbatch sweep"
echo "  2) small   $SIZES_SMALL"
echo "  3) large   $SIZES_LARGE"
echo "  4) custom  (type your own, semicolon-separated; spaces = extra layers)"
ask SIZE "Which sizes?" "1"
case "$SIZE" in
    1|full)   ARCHS="$SIZES_FULL"  ;;
    2|small)  ARCHS="$SIZES_SMALL" ;;
    3|large)  ARCHS="$SIZES_LARGE" ;;
    4|custom) ARCHS=""; ask ARCHS "Sizes, e.g. '16;128 128'" "$SIZES_FULL" ;;
    *)        ARCHS="$SIZE" ;;   # anything else is taken as a literal list
esac
IFS=';' read -r -a ARCH_LIST <<< "$ARCHS"

# Every entry must be layer widths, so a typo at the prompt above fails here
# instead of reaching main.py as a bogus --actor-fc-dim.
for arch in "${ARCH_LIST[@]}"; do
    if ! [[ "$arch" =~ ^[[:space:]]*[1-9][0-9]*([[:space:]]+[1-9][0-9]*)*[[:space:]]*$ ]]; then
        echo "error: '$arch' is not a valid layer spec; expected widths like '16' or '64 64'" >&2
        exit 1
    fi
done

# --- 4. the rest of the sweep --------------------------------------------
ask SEEDS "Which seeds (space-separated)?" "1 2 3"
ask NUM_WORKERS "Workers per run?" "3"
ask PROJECT "WandB project?" "aeos"
read -r -a SEED_LIST <<< "$SEEDS"

PER_ENV=$(( ${#ARCH_LIST[@]} * ${#SEED_LIST[@]} ))
TOTAL=$(( PER_ENV * ${#ENV_LIST[@]} ))
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ===========================================================================
# CLUSTER MODE — one chained sbatch per env, exactly like launch_all.sh
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

    echo
    echo "--- plan -------------------------------------------------------------"
    echo "  mode       : cluster (sbatch)"
    echo "  envs       : ${ENV_LIST[*]}   (one chained job each)"
    echo "  partition  : $PARTITION   account: $ACCOUNT"
    echo "  resources  : ${GPUS_PER_NODE} gpu(s), ${CPUS_PER_TASK} cpus, $MEM, $WALLTIME"
    echo "  sizes      : $ARCHS"
    echo "  seeds      : ${SEED_LIST[*]}"
    echo "  runs       : $PER_ENV per env ($TOTAL total), round-robin over $GPUS_PER_NODE gpu(s)"
    echo "  workers    : $NUM_WORKERS  (=> $(( PER_ENV * NUM_WORKERS )) worker processes per job)"
    echo "  chunks     : $NUM_CHUNKS per env"
    echo "  sbatch dir : $GEN_DIR"
    echo "----------------------------------------------------------------------"
    read -r -p "Submit? [y/N] " CONFIRM
    case "${CONFIRM:-n}" in
        y|Y|yes|YES) ;;
        *) echo "Aborted."; exit 0 ;;
    esac

    mkdir -p "$GEN_DIR"
    for env in "${ENV_LIST[@]}"; do
        sbatch_file="$GEN_DIR/aeose_${env}_${TIMESTAMP}.sbatch"
        {
            echo "#!/bin/bash"
            echo "#SBATCH --job-name=aeose_${env}"
            echo "#SBATCH --account=${ACCOUNT}"
            echo "#SBATCH --partition=${PARTITION}"
            echo "#SBATCH --nodes=1"
            echo "#SBATCH --gpus-per-node=${GPUS_PER_NODE}"
            echo "#SBATCH --cpus-per-task=${CPUS_PER_TASK}"
            echo "#SBATCH --mem=${MEM}"
            echo "#SBATCH --time=${WALLTIME}"
            echo "#SBATCH --output=aeose_${env}.o%j"
            echo "#SBATCH --mail-type=FAIL"
            echo "#SBATCH --mail-user=${MAIL_USER:-minjae5@illinois.edu}"
            echo
            echo "# Generated by scripts/launch_interactive.sh on $(date)"
            echo "source ~/.bashrc"
            echo "conda activate aeos"
            echo
            echo "export WANDB_INIT_TIMEOUT=300   # parallel launches init slowly"
            echo
            echo "ENV=${env}"
            echo
            i=0
            for arch in "${ARCH_LIST[@]}"; do
                read -r -a arch_dims <<< "$arch"
                for seed in "${SEED_LIST[@]}"; do
                    gpu=$(( i % GPUS_PER_NODE ))
                    echo "python3 main.py --project $PROJECT --env-name \$ENV --gpu-idx $gpu --num-workers $NUM_WORKERS --actor-fc-dim ${arch_dims[*]} --seed $seed &"
                    i=$(( i + 1 ))
                    # Stagger so concurrent WandB inits do not stampede.
                    if [ $(( i % GPUS_PER_NODE )) -eq 0 ]; then echo "sleep 10"; fi
                done
            done
            echo
            echo "wait"
        } > "$sbatch_file"
        chmod +x "$sbatch_file"

        echo
        echo "=== $env ==="
        echo "wrote $sbatch_file"
        bash "$SCRIPT_DIR/launch_chain.sh" "$sbatch_file" "$NUM_CHUNKS" \
            || echo "warning: failed to submit chain for '$env'; continuing with the rest" >&2
    done

    echo
    echo "All chains submitted. Inspect the queue with:  squeue -u \$USER"
    exit 0
fi

# ===========================================================================
# LOCAL MODE — same commands, no scheduler
# ===========================================================================
if command -v nvidia-smi >/dev/null 2>&1; then
    echo
    echo "Visible GPUs:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total \
               --format=csv,noheader || true
fi
echo
echo "Runs are spread round-robin over the GPUs you list (the sbatch scripts use two)."
ask GPUS "Which GPU index/indices (space-separated)?" "0 1"
read -r -a GPU_LIST <<< "$GPUS"
for g in "${GPU_LIST[@]}"; do
    if ! [[ "$g" =~ ^[0-9]+$ ]]; then
        echo "error: GPU index must be a non-negative integer, got '$g'" >&2
        exit 1
    fi
done

LOG_DIR="$REPO_ROOT/log/interactive/${TIMESTAMP}"

echo
echo "--- plan -------------------------------------------------------------"
echo "  mode       : local"
echo "  envs       : ${ENV_LIST[*]}"
echo "  gpus       : ${GPU_LIST[*]}"
echo "  sizes      : $ARCHS"
echo "  seeds      : ${SEED_LIST[*]}"
echo "  runs       : $PER_ENV per env, $TOTAL total"
echo "  workers    : $NUM_WORKERS  (=> $(( TOTAL * NUM_WORKERS )) worker processes total)"
echo "  stdout logs: $LOG_DIR"
echo "----------------------------------------------------------------------"
if [ "$TOTAL" -gt 18 ]; then
    echo "note: $TOTAL concurrent runs is more than one sbatch job's worth — make sure"
    echo "      this machine has the GPU memory and cores for it."
fi
read -r -p "Launch? [y/N] " CONFIRM
case "${CONFIRM:-n}" in
    y|Y|yes|YES) ;;
    *) echo "Aborted."; exit 0 ;;
esac

mkdir -p "$LOG_DIR"
export WANDB_INIT_TIMEOUT=${WANDB_INIT_TIMEOUT:-300}  # parallel launches init slowly

PIDS=()
i=0
for env in "${ENV_LIST[@]}"; do
    for arch in "${ARCH_LIST[@]}"; do
        read -r -a arch_dims <<< "$arch"
        tag=$(arch_tag "${arch_dims[@]}")
        for seed in "${SEED_LIST[@]}"; do
            gpu="${GPU_LIST[$(( i % ${#GPU_LIST[@]} ))]}"
            log_file="$LOG_DIR/${env}_fc${tag}_seed${seed}_gpu${gpu}.log"
            python3 main.py \
                --project "$PROJECT" \
                --env-name "$env" \
                --gpu-idx "$gpu" \
                --num-workers "$NUM_WORKERS" \
                --actor-fc-dim "${arch_dims[@]}" \
                --seed "$seed" \
                > "$log_file" 2>&1 &
            PIDS+=($!)
            echo "launched pid $! : env=$env fc=[${arch_dims[*]}] seed=$seed gpu=$gpu"
            i=$(( i + 1 ))
            # Stagger so concurrent WandB inits do not stampede.
            if [ $(( i % ${#GPU_LIST[@]} )) -eq 0 ]; then sleep 10; fi
        done
    done
done

echo
echo "$TOTAL run(s) launched. Tail them with:"
echo "  tail -f $LOG_DIR/*.log"
echo "Waiting for all runs (Ctrl-C detaches this shell; runs keep going)..."
wait "${PIDS[@]}"
echo "All runs finished."
