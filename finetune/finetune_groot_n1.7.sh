#!/usr/bin/env bash
#
# Fine-tune GR00T N1.7 on the SO-101 Bench teleop dataset with LeRobot's
# `lerobot-train` (GrootPolicy). Produces a **LeRobot-format** checkpoint.
#
# Run this from inside the LeRobot (grootn1.7) checkout, with the `groot` +
# `training` extras installed and `lerobot-train` on PATH. See finetune/README.md
# for prerequisites, hyperparameter notes, VRAM/multi-GPU guidance, and how to
# evaluate the result.
#
# Adapted from the official SO-101 recipe in the LeRobot docs
# (docs/source/groot.mdx) — new_embodiment, chunk_size=16, relative actions with
# the gripper excluded — retargeted at the SO-101 Bench dataset.
#
# Every knob below is overridable from the environment, e.g.:
#   STEPS=30000 BATCH_SIZE=32 ./finetune/finetune_groot_n1.7.sh
#
# For multi-GPU, don't call this directly — use finetune_groot_n1.7_multigpu.sh,
# which sets LERO_LAUNCH to an `accelerate launch ...` prefix consumed below.

set -euo pipefail

# Launcher: `lerobot-train` for single-GPU, or an `accelerate launch ... $(which
# lerobot-train)` prefix for multi-GPU (set by the multigpu wrapper).
LERO_LAUNCH="${LERO_LAUNCH:-lerobot-train}"

# --- dataset / model ---
DATASET="${DATASET:-5hadytru/so101_bench_sim_1}"        # SO-101 sim teleop set (480 eps, front+overhead)
# Pin the branch: this dataset has no LeRobot `codebase_version` git tag, so the default
# revision lookup fails with RevisionNotFoundError. `main` points straight at the data.
DATASET_REVISION="${DATASET_REVISION:-main}"
BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.7-3B}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-new_embodiment}"       # SO-101 uses the new_embodiment head

# --- action chunk (keep in sync with groot_eval --action_horizon) ---
CHUNK_SIZE="${CHUNK_SIZE:-16}"

# --- optimization ---
BATCH_SIZE="${BATCH_SIZE:-8}"                            # 3B model — single-GPU-safe default; raise if VRAM allows
STEPS="${STEPS:-20000}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
SEED="${SEED:-42}"

# --- output / logging ---
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/so101_bench_groot_n1.7}"
JOB_NAME="${JOB_NAME:-so101_bench_groot_n1.7}"
WANDB_ENABLE="${WANDB_ENABLE:-true}"

# --- optional push to the Hub ---
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"
REPO_ID="${REPO_ID:-}"   # required when PUSH_TO_HUB=true

# --- relative actions ---
# The SO-101 groot.mdx recipe predicts joint DELTAS (gripper absolute). The N1.7 checkout
# supports these flags (the preflight below guarantees we're on it). Set RELATIVE_ACTIONS=false
# to train absolute actions instead.
RELATIVE_ACTIONS="${RELATIVE_ACTIONS:-true}"

if [[ "$PUSH_TO_HUB" == "true" && -z "$REPO_ID" ]]; then
  echo "PUSH_TO_HUB=true requires REPO_ID=<hf_user>/<name>" >&2
  exit 1
fi

# Preflight: this recipe targets GR00T N1.7 (Cosmos-Reason2 / Qwen3-VL). The `lerobot`
# on PATH must be the N1.7 implementation (module `groot_n1_7`), NOT an older N1.5 build
# (Eagle backbone, `groot_n1`), which cannot load nvidia/GR00T-N1.7-3B.
if [[ "${SKIP_VERSION_CHECK:-0}" != "1" ]]; then
  if ! python -c "import lerobot.policies.groot.groot_n1_7" 2>/dev/null; then
    echo "ERROR: the active 'lerobot' does not provide GR00T N1.7 (no groot_n1_7 module)." >&2
    echo "       It looks like an N1.5 build (Eagle backbone), which can't fine-tune N1.7." >&2
    echo "       Install / activate the N1.7 checkout in its own env, e.g.:" >&2
    echo "         cd ../grootn1.7/lerobot && pip install -e '.[groot,training]'" >&2
    echo "       (a fresh env — its transformers 5.x pin conflicts with N1.5's Eagle stack)." >&2
    echo "       Then re-run this script. Override with SKIP_VERSION_CHECK=1 (not advised)." >&2
    exit 1
  fi
fi

# Build the arg list as an array so conditional flags and JSON quoting stay correct.
ARGS=(
  --dataset.repo_id="$DATASET"
  --dataset.revision="$DATASET_REVISION"
  --dataset.image_transforms.enable=true
  --policy.type=groot
  --policy.device=cuda
  --policy.base_model_path="$BASE_MODEL"
  --policy.embodiment_tag="$EMBODIMENT_TAG"
  --policy.chunk_size="$CHUNK_SIZE"
  --policy.n_action_steps="$CHUNK_SIZE"
  --policy.use_bf16=true
  --policy.push_to_hub="$PUSH_TO_HUB"
  --seed="$SEED"
  --batch_size="$BATCH_SIZE"
  --steps="$STEPS"
  --save_checkpoint=true
  --save_freq="$SAVE_FREQ"
  --use_policy_training_preset=true
  --env_eval_freq=0
  --eval_steps=0
  --log_freq=10
  --output_dir="$OUTPUT_DIR"
  --job_name="$JOB_NAME"
  --wandb.enable="$WANDB_ENABLE"
  --wandb.disable_artifact=true
)
if [[ "$RELATIVE_ACTIONS" == "true" ]]; then
  ARGS+=(--policy.use_relative_actions=true --policy.relative_exclude_joints='["gripper"]')
fi
if [[ -n "$REPO_ID" ]]; then
  ARGS+=(--policy.repo_id="$REPO_ID")
fi

set -x
# shellcheck disable=SC2086  # LERO_LAUNCH is an intentional multi-word launcher prefix
$LERO_LAUNCH "${ARGS[@]}"
