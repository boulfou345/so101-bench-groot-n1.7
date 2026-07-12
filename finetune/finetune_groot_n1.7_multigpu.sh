#!/usr/bin/env bash
#
# Multi-GPU wrapper around finetune_groot_n1.7.sh using HuggingFace `accelerate`.
#
# It only sets LERO_LAUNCH to an `accelerate launch` prefix and delegates to the
# single-GPU script, so all the same env-var knobs (STEPS, BATCH_SIZE, OUTPUT_DIR,
# PUSH_TO_HUB/REPO_ID, ...) apply unchanged.
#
#   NUM_GPUS=4 BATCH_SIZE=16 ./finetune/finetune_groot_n1.7_multigpu.sh
#
# Notes:
#   * --batch_size is PER-PROCESS in lerobot-train, so effective batch =
#     BATCH_SIZE * NUM_GPUS. Lower BATCH_SIZE accordingly.
#   * For FSDP / sharded training on the 3B backbone, pass an accelerate config:
#     ACCELERATE_ARGS="--config_file fsdp.yaml" NUM_GPUS=4 ./...multigpu.sh
#   * See docs/source/multi_gpu_training.mdx in the LeRobot (grootn1.7) checkout.
#   * With accelerate, mixed precision is controlled by the accelerate config, not
#     by --policy.use_amp.

set -euo pipefail

NUM_GPUS="${NUM_GPUS:-2}"
ACCELERATE_ARGS="${ACCELERATE_ARGS:-}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LERO_TRAIN="$(command -v lerobot-train || true)"
if [[ -z "$LERO_TRAIN" ]]; then
  echo "lerobot-train not found on PATH. Install the LeRobot checkout with the" >&2
  echo "  [groot,training] extras and activate its environment first." >&2
  exit 1
fi

# shellcheck disable=SC2086  # ACCELERATE_ARGS is an intentional multi-word prefix
export LERO_LAUNCH="accelerate launch $ACCELERATE_ARGS --num_processes=$NUM_GPUS $LERO_TRAIN"

exec "$HERE/finetune_groot_n1.7.sh"
