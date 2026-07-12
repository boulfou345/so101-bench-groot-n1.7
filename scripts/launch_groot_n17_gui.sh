#!/usr/bin/env bash
# Single-command GR00T N1.7 eval with the Isaac Sim GUI.
#
# Starts the ZMQ bridge (finetune/zmq_bridge_server.py) serving the fine-tuned
# N1.7 LeRobot checkpoint on :5556 if it is not already up, waits for it, then
# launches groot_eval.py against it. Extra flags pass through to groot_eval.py:
#
#   ./scripts/launch_groot_n17_gui.sh --num_episodes 5
#
# The bridge is left running afterwards (checkpoint load is slow); stop it with:
#   pkill -f zmq_bridge_server.py
set -euo pipefail

cd "$(dirname "$0")/.."

N17_VENV_PY="${N17_VENV_PY:-$HOME/Documents/medium_blog/grootn1.7/lerobot/.venv/bin/python}"
# Unique per run: two evals sharing a repo_root crash the recorder on the
# partially-written dataset. Set REPO_ROOT explicitly to resume a finished one.
REPO_ROOT="${REPO_ROOT:-data/lerobot/groot_n17_gui_$(date +%Y%m%d_%H%M%S)}"
N17_CKPT="${N17_CKPT:-outputs/train/so101_bench_groot_n1.7/checkpoints/last/pretrained_model}"
BRIDGE_PORT="${BRIDGE_PORT:-5556}"
BRIDGE_LOG="${BRIDGE_LOG:-/tmp/groot_n17_bridge.log}"

if pgrep -f "groot_eval.py" >/dev/null; then
  echo "[launch] WARNING: another groot_eval.py is already running (GPU contention likely):"
  pgrep -af "groot_eval.py" | grep python3 | head -2
fi

if ! ss -ltn | grep -q ":${BRIDGE_PORT} "; then
  echo "[launch] starting N1.7 bridge on :${BRIDGE_PORT} (log: ${BRIDGE_LOG})"
  PYTHONUNBUFFERED=1 nohup "${N17_VENV_PY}" finetune/zmq_bridge_server.py \
    --model-path "${N17_CKPT}" \
    --host 0.0.0.0 --port "${BRIDGE_PORT}" --action-horizon 16 \
    > "${BRIDGE_LOG}" 2>&1 &
  echo "[launch] waiting for bridge (checkpoint load takes a minute)..."
  until grep -q "serving on" "${BRIDGE_LOG}" 2>/dev/null; do
    if grep -qE "Traceback|Error" "${BRIDGE_LOG}" 2>/dev/null; then
      echo "[launch] bridge failed to start:" >&2
      tail -20 "${BRIDGE_LOG}" >&2
      exit 1
    fi
    sleep 2
  done
fi
echo "[launch] bridge serving on :${BRIDGE_PORT}"

PYTHONUNBUFFERED=1 ~/IsaacLab/isaaclab.sh -p scripts/groot_eval.py \
  --task So101Bench-Bin-v0 \
  --episodes_jsonl tasks/real_gr00t_WM_combined.jsonl \
  --policy_host localhost \
  --policy_port "${BRIDGE_PORT}" \
  --action_horizon 16 \
  --use_overhead_init false \
  --record_dataset \
  --repo_root "${REPO_ROOT}" \
  "$@"
