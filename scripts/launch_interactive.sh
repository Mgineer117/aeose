#!/bin/bash
# ---------------------------------------------------------------------------
# Interactive front-end for the same sweep launch_all.sh submits.
#
# launch_all.sh submits scripts/run_aeose_<env>.sbatch for every env, each of
# which runs the fixed 18-run sweep (6 actor sizes x 3 seeds, split over 2
# GPUs). This script runs that same sweep — sizes and seeds are fixed, not
# prompted for — and asks only where to run it: which envs, which GPUs, and
# which cluster resources.
#
#   cluster : generates an sbatch script per env and submits it, after letting
#             you pick a partition from the ones that have free GPUs. A run
#             needs about 7-00:00:00; partitions capped below that are
#             submitted in dependency mode — a checkpoint-restart chain via
#             launch_chain.sh, as launch_all.sh does — while partitions that
#             allow a full week get a single job.
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
#   MODE=cluster ENVS="charge desat" PARTITION=gpuA100x8 \
#       bash scripts/launch_interactive.sh
#
# The sweep itself is fixed, but ARCHS= and SEEDS= override it if needed:
#   ARCHS="64 64;256 256" SEEDS="1 2" bash scripts/launch_interactive.sh
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

# A run needs about a week of wall clock. Partitions capped below that must be
# chained (checkpoint-restart), which is what launch_chain.sh sets up.
FULL_RUN="7-00:00:00"

# to_secs "2-00:00:00" -> 172800.  Accepts the SLURM time formats
# d-hh:mm:ss, d-hh:mm, d-hh, hh:mm:ss and mm:ss. Echoes -1 for an unlimited
# partition and 0 when the value cannot be parsed.
to_secs() {
    local t=$1 days=0 rest
    case "$t" in
        infinite|INFINITE|unlimited|UNLIMITED) echo -1; return ;;
        "" | n/a | N/A) echo 0; return ;;
    esac
    if [[ "$t" == *-* ]]; then
        days=${t%%-*}
        rest=${t#*-}
    else
        rest=$t
    fi
    local IFS=: parts
    read -r -a parts <<< "$rest"
    local h=0 m=0 s=0
    case ${#parts[@]} in
        3) h=${parts[0]}; m=${parts[1]}; s=${parts[2]} ;;
        2) if [[ "$t" == *-* ]]; then h=${parts[0]}; m=${parts[1]}; else m=${parts[0]}; s=${parts[1]}; fi ;;
        1) if [[ "$t" == *-* ]]; then h=${parts[0]}; else m=${parts[0]}; fi ;;
        *) echo 0; return ;;
    esac
    local f
    for f in "$days" "$h" "$m" "$s"; do
        [[ "$f" =~ ^[0-9]+$ ]] || { echo 0; return; }
    done
    # Strip leading zeros so 08 is not read as invalid octal.
    days=$((10#$days)); h=$((10#$h)); m=$((10#$m)); s=$((10#$s))
    echo $(( days * 86400 + h * 3600 + m * 60 + s ))
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

# --- 3. the sweep itself (fixed: full size sweep x 3 seeds) --------------
# Not prompted for — every launch runs the same sweep the run_aeose_*.sbatch
# scripts do. Override with ARCHS= / SEEDS= if you ever need a subset.
ARCHS="${ARCHS:-$SIZES_FULL}"
SEEDS="${SEEDS:-1 2 3}"
IFS=';' read -r -a ARCH_LIST <<< "$ARCHS"
read -r -a SEED_LIST <<< "$SEEDS"

# Catch a malformed ARCHS override here rather than in main.py's --actor-fc-dim.
for arch in "${ARCH_LIST[@]}"; do
    if ! [[ "$arch" =~ ^[[:space:]]*[1-9][0-9]*([[:space:]]+[1-9][0-9]*)*[[:space:]]*$ ]]; then
        echo "error: '$arch' is not a valid layer spec; expected widths like '16' or '64 64'" >&2
        exit 1
    fi
done

echo
echo "Sweep: sizes $ARCHS"
echo "       seeds $SEEDS"

ask NUM_WORKERS "Workers per run?" "3"
ask PROJECT "WandB project?" "aeos"

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
    FREE_TABLE=$(sinfo -h -N -O 'Partition:40,StateLong:20,Gres:60,GresUsed:60,Time:20' 2>/dev/null \
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
            part = $1; sub(/\*$/, "", part)     # strip default-partition marker
            # SLURM appends flag characters to the state: mixed$ (reserved),
            # idle~ (powered down), idle* (unresponsive), mixed# (booting).
            # Matching "idle"/"mixed" exactly would hide every flagged node.
            state = $2; sub(/[*~#!%$@+]+$/, "", state)
            if (state ~ /^(down|drain|drng|fail|error|maint|unknown|future|invalid|reserved|boot|power|planned)/) next
            tot = gpus($3)
            if (tot <= 0) next                  # not a GPU node
            free = tot - gpus($4)
            if (free < 0) free = 0
            gpu_total[part] += tot
            gpu_free[part] += free
            limit[part] = $5
            if (free > 0) nodes[part] += 1
        }
        END {
            # FREE rows become the menu; BUSY rows answer "why is my partition
            # missing?" — it has GPUs, they are just all allocated right now.
            for (p in gpu_total) {
                if (gpu_free[p] > 0)
                    printf "FREE %s %d %d %s\n", p, gpu_free[p], nodes[p], limit[p]
                else
                    printf "BUSY %s %d %s\n", p, gpu_total[p], limit[p]
            }
        }
      ') || true
    BUSY_TABLE=$(printf '%s\n' "$FREE_TABLE" | awk '$1 == "BUSY" {print $2}' | sort | tr '\n' ' ')
    FREE_TABLE=$(printf '%s\n' "$FREE_TABLE" | awk '$1 == "FREE" {$1 = ""; sub(/^ /, ""); print}' | sort -k2 -nr)

    if [ -n "$FREE_TABLE" ]; then
        echo
        printf "  %-3s %-24s %-10s %-15s %s\n" "#" "PARTITION" "FREE GPUS" "IDLE/MIX NODES" "TIME LIMIT"
        PART_NAMES=()
        PART_LIMITS=()
        while read -r p free nodes limit; do
            PART_NAMES+=("$p")
            PART_LIMITS+=("$limit")
            printf "  %-3s %-24s %-10s %-15s %s\n" "${#PART_NAMES[@]}" "$p" "$free" "$nodes" "$limit"
        done <<< "$FREE_TABLE"
        if [ -n "$BUSY_TABLE" ]; then
            echo
            echo "  GPU partitions with none free right now: $BUSY_TABLE"
        fi
        echo
        echo "Pick a number from the list, or type a partition name directly"
        echo "(any partition works, including a busy one — you just queue)."
        ask PARTITION "Which partition?" "${PART_NAMES[0]}"
        # A bare number selects from the table above.
        if [[ "$PARTITION" =~ ^[0-9]+$ ]]; then
            if [ "$PARTITION" -lt 1 ] || [ "$PARTITION" -gt "${#PART_NAMES[@]}" ]; then
                echo "error: choice '$PARTITION' is out of range 1-${#PART_NAMES[@]}" >&2
                exit 1
            fi
            PART_LIMIT="${PART_LIMITS[$((PARTITION - 1))]}"
            PARTITION="${PART_NAMES[$((PARTITION - 1))]}"
            echo "  -> $PARTITION"
        else
            # Typed a name (or took the default): reuse the table's limit when
            # it is one of the listed partitions, rather than re-querying.
            for idx in "${!PART_NAMES[@]}"; do
                if [ "${PART_NAMES[$idx]}" = "$PARTITION" ]; then
                    PART_LIMIT="${PART_LIMITS[$idx]}"
                    break
                fi
            done
        fi
    else
        echo "  (no idle/mixed nodes with free GPUs right now, or sinfo lacks"
        echo "   GresUsed support — falling back to a plain partition list)"
        sinfo -h -o '  %P  avail=%a  nodes=%D  state=%T' 2>/dev/null | sort -u || true
        ask PARTITION "Which partition?" "gpuA100x8"
    fi

    # --- time limit decides whether we chain ------------------------------
    # Not every path above knows the limit (typed-in name, fallback listing),
    # so ask SLURM directly when it is still unset.
    if [ -z "${PART_LIMIT:-}" ]; then
        PART_LIMIT=$(sinfo -h -p "$PARTITION" -O 'Time:20' 2>/dev/null | head -1 | tr -d ' ') || true
    fi
    LIMIT_SECS=$(to_secs "${PART_LIMIT:-}")
    FULL_SECS=$(to_secs "$FULL_RUN")

    echo
    if [ "$LIMIT_SECS" -eq 0 ]; then
        echo "Could not read a time limit for '$PARTITION' — treating it as capped."
        DEPENDENCY_MODE=1
        PART_LIMIT="${PART_LIMIT:-unknown}"
    elif [ "$LIMIT_SECS" -lt 0 ] || [ "$LIMIT_SECS" -ge "$FULL_SECS" ]; then
        # Room for a full run in one job, so no checkpoint-restart chain needed.
        echo "Partition '$PARTITION' allows $PART_LIMIT (>= $FULL_RUN): single job, no dependency chain."
        DEPENDENCY_MODE=0
    else
        echo "Partition '$PARTITION' caps jobs at $PART_LIMIT (< $FULL_RUN):"
        echo "  -> dependency mode; chunks are chained with afterany so each one"
        echo "     resumes the previous chunk's checkpoint."
        DEPENDENCY_MODE=1
    fi

    ask ACCOUNT "Which account?" "bhqw-delta-gpu"
    ask GPUS_PER_NODE "GPUs per node to request?" "2"
    ask CPUS_PER_TASK "CPUs per task?" "32"
    ask MEM "Memory?" "240G"

    if [ "$DEPENDENCY_MODE" -eq 1 ]; then
        # Ask for the partition cap by default, then chain enough chunks to add
        # up to a full run.
        default_wall=$PART_LIMIT
        [ "$default_wall" = unknown ] && default_wall="2-00:00:00"
        ask WALLTIME "Wall time per chunk?" "$default_wall"
        wall_secs=$(to_secs "$WALLTIME")
        if [ "$wall_secs" -gt 0 ]; then
            default_chunks=$(( (FULL_SECS + wall_secs - 1) / wall_secs ))
        else
            default_chunks=3
        fi
        [ "$default_chunks" -lt 1 ] && default_chunks=1
        ask NUM_CHUNKS "How many chained chunks (~$FULL_RUN total)?" "$default_chunks"
    else
        ask WALLTIME "Wall time?" "$FULL_RUN"
        NUM_CHUNKS=${NUM_CHUNKS:-1}
    fi

    if ! [[ "$GPUS_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
        echo "error: --gpus-per-node must be a positive integer, got '$GPUS_PER_NODE'" >&2
        exit 1
    fi
    if ! [[ "$NUM_CHUNKS" =~ ^[1-9][0-9]*$ ]]; then
        echo "error: chunks must be a positive integer, got '$NUM_CHUNKS'" >&2
        exit 1
    fi
    # Refuse a wall time the partition will reject outright.
    wall_secs=$(to_secs "$WALLTIME")
    if [ "$LIMIT_SECS" -gt 0 ] && [ "$wall_secs" -gt "$LIMIT_SECS" ]; then
        echo "error: wall time $WALLTIME exceeds the $PARTITION limit of $PART_LIMIT" >&2
        exit 1
    fi

    GEN_DIR="$REPO_ROOT/scripts/generated"

    echo
    echo "--- plan -------------------------------------------------------------"
    if [ "$DEPENDENCY_MODE" -eq 1 ]; then
        submission="dependency chain, $NUM_CHUNKS chunk(s) per env"
    else
        submission="single job per env"
    fi
    echo "  mode       : cluster (sbatch)"
    echo "  envs       : ${ENV_LIST[*]}"
    echo "  submission : $submission"
    echo "  partition  : $PARTITION (limit $PART_LIMIT)   account: $ACCOUNT"
    echo "  resources  : ${GPUS_PER_NODE} gpu(s), ${CPUS_PER_TASK} cpus, $MEM, $WALLTIME per job"
    echo "  sizes      : $ARCHS"
    echo "  seeds      : ${SEED_LIST[*]}"
    echo "  runs       : $PER_ENV per env ($TOTAL total), round-robin over $GPUS_PER_NODE gpu(s)"
    echo "  workers    : $NUM_WORKERS  (=> $(( PER_ENV * NUM_WORKERS )) worker processes per job)"
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
        # launch_chain.sh with 1 chunk is a plain sbatch, with N a dependency
        # chain — so the same call covers both modes.
        bash "$SCRIPT_DIR/launch_chain.sh" "$sbatch_file" "$NUM_CHUNKS" \
            || echo "warning: failed to submit '$env'; continuing with the rest" >&2
    done

    echo
    echo "All jobs submitted. Inspect the queue with:  squeue -u \$USER"
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
            # bash already ignores SIGINT in async children of a non-interactive
            # script, so Ctrl-C alone does not kill a run. Ignoring HUP too (the
            # disposition survives exec) additionally keeps runs alive when the
            # terminal or ssh session goes away, and </dev/null stops a run from
            # blocking on terminal input.
            (
                trap '' INT HUP
                exec python3 main.py \
                    --project "$PROJECT" \
                    --env-name "$env" \
                    --gpu-idx "$gpu" \
                    --num-workers "$NUM_WORKERS" \
                    --actor-fc-dim "${arch_dims[@]}" \
                    --seed "$seed" \
                    < /dev/null > "$log_file" 2>&1
            ) &
            PIDS+=($!)
            echo "launched pid $! : env=$env fc=[${arch_dims[*]}] seed=$seed gpu=$gpu"
            i=$(( i + 1 ))
            # Stagger so concurrent WandB inits do not stampede.
            if [ $(( i % ${#GPU_LIST[@]} )) -eq 0 ]; then sleep 10; fi
        done
    done
done

echo
echo "$TOTAL run(s) launched. Useful commands:"
echo "  tail -f $LOG_DIR/*.log"
echo "  pkill -f 'main.py --project $PROJECT'   # stop every run"
echo
echo "Ctrl-C now only detaches this launcher; the runs keep going."
trap 'echo; echo "Detached. Runs continue in the background; logs in $LOG_DIR"; exit 0' INT
wait "${PIDS[@]}"
echo "All runs finished."
