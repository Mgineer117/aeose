#!/bin/bash
# ---------------------------------------------------------------------------
# Interactive smoke test for the checkpoint-restart resume path (PPO).
#
# Run it from the repo root WITH THE CONDA ENV ACTIVE:
#     conda activate aeos
#     bash scripts/test_resume.sh
#
# It does NOT touch your real logs or W&B:
#   - logs to an isolated dir (log/test_resume_log/, wiped at start)
#   - forces WANDB_MODE=offline
#
# What it checks, end to end, in ~a few minutes:
#   1. Phase 1 trains a tiny run and writes latest_model.pth + resume_state.json.
#   2. latest_model.pth is a FULL training checkpoint (actor + critic +
#      optimizer + obs_rms), not just the actor.
#   3. Phase 2 (same env/arch/seed, larger --timesteps) auto-detects the
#      checkpoint, prints "Resuming from latest checkpoint", and continues
#      training from phase 1's timestep instead of restarting at 0.
#
# All sizes are tiny and overridable from the environment, e.g.:
#     T1=200 T2=400 ENV=charge ARCH="4" bash scripts/test_resume.sh
# Lower T1/T2 if your env steps are slow; raise them if a phase does too
# little to produce a checkpoint.
# ---------------------------------------------------------------------------
set -euo pipefail

# --- knobs (override via env) ---
ENV=${ENV:-charge}            # charge | resource | desat | downlink
ARCH=${ARCH:-4}               # actor hidden dims (space-separated if multi-layer)
SEED=${SEED:-1}
T1=${T1:-300}                 # phase-1 timesteps (must be > one batch)
T2=${T2:-600}                 # phase-2 timesteps (must be > T1 to force continued training)

LOGROOT="log/test_resume_log"
COMMON_ARGS=(
    --project aeos_resume_test
    --logdir "$LOGROOT"
    --env-name "$ENV"
    --actor-fc-dim $ARCH
    --seed "$SEED"
    --num-runs 1
    --num-workers 1
    --eval-num 1
    --log-interval 2
    --minibatch-size 64
    --num-minibatch 1
    --batch-size 64
    --checkpoint-interval-sec 5
)

export WANDB_MODE=offline
export WANDB_SILENT=true

red()   { printf "\033[31m%s\033[0m\n" "$*"; }
green() { printf "\033[32m%s\033[0m\n" "$*"; }
bold()  { printf "\033[1m%s\033[0m\n" "$*"; }

fail() { red "FAIL: $*"; exit 1; }

# Read {"step": N} from the single resume_state.json under LOGROOT.
read_step() {
    python3 - "$LOGROOT" <<'PY'
import glob, json, sys
hits = glob.glob(sys.argv[1] + "/**/resume_state.json", recursive=True)
if not hits:
    print("MISSING"); sys.exit(0)
with open(hits[0]) as f:
    print(int(json.load(f).get("step", -1)))
PY
}

bold "== Cleaning $LOGROOT =="
rm -rf "$LOGROOT"
mkdir -p "$LOGROOT"

# ------------------------------------------------------------------ Phase 1
bold "== Phase 1: fresh training to ~$T1 timesteps =="
python3 main.py "${COMMON_ARGS[@]}" --timesteps "$T1" 2>&1 | tee "$LOGROOT/phase1.out"

CKPT=$(find "$LOGROOT" -name latest_model.pth | head -1 || true)
[ -n "$CKPT" ] || fail "phase 1 did not produce latest_model.pth"
green "phase 1 wrote checkpoint: $CKPT"

STEP1=$(read_step)
[ "$STEP1" != "MISSING" ] || fail "phase 1 did not write resume_state.json"
green "phase 1 resume_state step = $STEP1"

# Verify the checkpoint is FULL training state, not actor-only.
bold "== Verifying full-state checkpoint contents =="
python3 - "$CKPT" <<'PY'
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu")
if not isinstance(ck, dict):
    raise SystemExit(f"FAIL: checkpoint is {type(ck)}, expected a dict of components")
required = ["actor", "critic", "optimizer"]
missing = [k for k in required if k not in ck]
if missing:
    raise SystemExit(f"FAIL: checkpoint missing {missing}; keys present: {list(ck.keys())}")
print("OK: checkpoint keys =", list(ck.keys()))
PY
green "full-state checkpoint verified (actor + critic + optimizer [+ obs_rms])"

# ------------------------------------------------------------------ Phase 2
bold "== Phase 2: same identity, --timesteps $T2 (should resume + continue) =="
python3 main.py "${COMMON_ARGS[@]}" --timesteps "$T2" 2>&1 | tee "$LOGROOT/phase2.out"

grep -q "Resuming from latest checkpoint" "$LOGROOT/phase2.out" \
    || fail "phase 2 did NOT resume (no 'Resuming from latest checkpoint' message)"
green "phase 2 resumed from the phase-1 checkpoint"

STEP2=$(read_step)
[ "$STEP2" != "MISSING" ] || fail "phase 2 lost resume_state.json"
python3 -c "import sys; sys.exit(0 if $STEP2 > $STEP1 else 1)" \
    || fail "phase 2 step ($STEP2) did not advance past phase 1 step ($STEP1)"
green "phase 2 advanced the timestep: $STEP1 -> $STEP2"

echo
green "===================================================="
green " PASS: resume works — full-state checkpoint, detected"
green "       automatically, and training continued ($STEP1 -> $STEP2)."
green "===================================================="
