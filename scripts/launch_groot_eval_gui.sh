#!/usr/bin/env bash
# Launch the GR00T evaluator with the Isaac Sim GUI (no --headless).
# Assumes the GR00T policy server is already running on localhost:5555.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHONUNBUFFERED=1 ~/IsaacLab/isaaclab.sh -p scripts/groot_eval.py \
  --task So101Bench-Bin-v0 \
  --episodes_jsonl tasks/real_gr00t_WM_combined.jsonl \
  --policy_host localhost \
  --policy_port 5555 \
  --action_horizon 16 \
  --use_overhead_init true \
  --record_dataset \
  --repo_root data/lerobot/groot_n16_real_sim_1_ah16 \
  "$@"
