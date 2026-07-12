# SO-101 Bench × GR00T N1.7 — Eval run report

**Date:** 2026-07-08
**Checkpoint:** `outputs/train/so101_bench_groot_n1.7/checkpoints/last` (→ `020000`, LeRobot `GrootPolicy`, train loss 0.065)
**Task:** `So101Bench-Bin-v0` · `tasks/custom_bin.jsonl` (5 episodes)
**Serving path:** LeRobot checkpoint → `finetune/zmq_bridge_server.py` (ZMQ bridge, port **5556**) → `scripts/groot_eval.py` (Isaac Lab, `--headless`)

---

## TL;DR

The full pipeline ran **end-to-end**: the fine-tuned N1.7 checkpoint was served over the ZMQ
bridge, Isaac Lab drove all 5 `custom_bin` episodes, the policy was queried live, and every
episode was scored.

**Result: 0 / 5 success (0.0%). All 5 episodes ended in `time_out`.** The policy connected
and inferred without a single error, but the arm did not complete any bin placement within
the episode time limit.

---

## Results

### Per-episode outcomes

| # | Active object(s) | Instruction | Success | End reason | Sim length |
|---|---|---|:---:|---|---:|
| 1 | `object_1` — black pen | Place each object in the plastic bin | ❌ | `time_out` | 25.0 s |
| 2 | `object_2` — red tape | Place each object in the plastic bin | ❌ | `time_out` | 25.0 s |
| 3 | `object_3` — pink eraser | Place each object in the plastic bin | ❌ | `time_out` | 25.0 s |
| 4 | `object_4` — blue scissors | Place each object in the plastic bin | ❌ | `time_out` | 25.0 s |
| 5 | all 4 (pen, tape, eraser, scissors) | Place each object in the plastic bin | ❌ | `time_out` | 90.0 s |

### Aggregate (from `groot_eval.py`)

```
Success Rate: 0/5 (0.0%), skipped=0
Episode end reasons: time_out=5
Postmortem failure types (5 failed episode(s)): not_applicable=5 (100.0%)
```

- **`time_out` on every episode** — no episode was terminated early by a success or a
  live-failure condition; each ran to its full step budget (single-object 25 s, 4-object 90 s).
- **`live_failure_reason=none`** for all — no in-episode failure trigger (e.g. object knocked
  off table) fired.
- **Postmortem `not_applicable=5`** — the scorer found no classifiable failure mode
  (mis-grasp, wrong object, etc.), consistent with the arm not making a scorable attempt
  rather than attempting and failing.

---

## Pipeline-stage status (all green)

| Stage | Result | Evidence |
|---|---|---|
| Checkpoint self-test (load + infer) | ✅ | `images=['front','overhead'] action_dim=6 relative=True`; `(16,5)`+`(16,1)` chunk |
| Bridge serving over ZMQ | ✅ | `serving on tcp://0.0.0.0:5556`; **0** inference errors across the whole run |
| Isaac Sim scene / env bring-up | ✅ | scene created 0.88 s; 4 objects pre-spawned; 2 cameras found |
| Policy connection | ✅ | `Connecting to GR00T policy server at localhost:5556… Policy server connected.` |
| Camera mapping | ✅ | `{'wrist': 'front', 'overhead': 'overhead'}` (matches the checkpoint's 2 cameras) |
| Rollout + scoring | ✅ | all 5 episodes stepped and scored |
| **Task success** | ❌ | 0 / 5 — see interpretation below |

---

## Root cause of the two earlier "no-result" runs (resolved)

The first two launches appeared to "hang ~10 min then exit 0 with no results." That was a
**diagnostics artifact, not a real failure**:

1. **`groot_eval.py` stdout was block-buffered.** All `[INFO]` progress/result prints buffer
   when piped to a file; Isaac Sim's shutdown does a hard exit that skips the flush, so the
   entire `[INFO]` stream (including per-episode outcomes) was discarded. Only Isaac's
   stderr warnings survived — hence the illusion of a silent 10-minute gap.
2. **The bridge logs nothing on success.** Its serve loop prints only on startup and on
   *errors*, so "zero bridge requests" was never real evidence — it was a signal the bridge
   doesn't emit.

**Fix:** re-run with `PYTHONUNBUFFERED=1`. The run then showed full live progress and the
0/5 summary. (The earlier GPU cleanup — reclaiming ~20 GB of stale Jul-05 `groot_eval`
processes and zombie CUDA contexts — was good hygiene but was **not** the blocker.)

---

## Interpretation — the arm moves and reaches, but misses the grasp

A recorded rollout (`--record_dataset`, 67 MB LeRobot dataset with wrist + overhead video
and per-frame state/action) settles what 0/5 means. **This is a genuine grasp-execution
failure, not a dead/frozen integration.** Evidence:

- **The arm moves substantially and tracks commands.** Per-episode joint range of motion is
  large (e.g. shoulder-lift 100–129°, wrist-roll up to 118°, gripper actuating), and
  recorded `action ≈ observation.state` range per joint — the arm faithfully executes the
  policy's commanded targets. So relative-action decode and state/action units are
  effectively correct (a gross offset or unit bug would not track cleanly across all 6 dims).
- **It reaches the right region.** Overhead frames show the end-effector extending out and
  down toward the target object (single-object episodes) and descending among the correct
  four objects in the 4-object episode.
- **It fails the fine grasp.** The gripper hovers near the object but never aligns/closes to
  lift it; the target stays on the table, nothing reaches the bin, and the episode times out
  with `not_applicable` postmortem (no scorable pick attempt completed).

This matches the benchmark's central thesis (semantic ≫ geometric competence): the policy
knows *where* to go but not *how* to grasp precisely. Remaining factors worth probing to
raise the number, roughly in priority order:

1. **Grasp precision / depth.** The end-effector approaches but misses — check wrist-camera
   framing and the approach/close timing vs. the training demos; a small vertical or
   depth-alignment bias would produce exactly this near-miss.
2. **More training / WM conditioning.** This is the 20k-step, 2-camera (no `overhead_init`)
   checkpoint. Longer training and/or the WM (`overhead_init`) variant may improve grasp
   precision (see `add_overhead_init.py`).
3. **Episode time budget.** 25 s single-object is enough to reach but tight for retries;
   a longer budget would reveal whether it eventually recovers a grasp.
4. **Sim-vs-teleop dynamics.** Confirm the sim reset pose, object scale, and gripper
   friction/contact match the teleop dataset the checkpoint learned from.

_(Overhead frames saved during analysis: t=1/8/16/24 s for episode 1, t=110 s for the
4-object episode — arm reaches toward objects but does not grasp.)_

---

## Reusable state

- **Bridge is still running** on `tcp://0.0.0.0:5556` (checkpoint already loaded, ~15.5 GB)
  — re-runs can point straight at `--policy_port 5556` with no reload.
- GPU has ~44 GB free after the cleanup; Isaac Sim eval fits comfortably.

## Reproduce / next run

```bash
# bridge already up on 5556; if not:
#   grootn1.7/lerobot/.venv/bin/python finetune/zmq_bridge_server.py \
#     --model-path outputs/train/so101_bench_groot_n1.7/checkpoints/last/pretrained_model \
#     --host 0.0.0.0 --port 5556 --action-horizon 16

# ALWAYS run unbuffered so results are visible:
PYTHONUNBUFFERED=1 /home/zeux/IsaacLab/isaaclab.sh -p scripts/groot_eval.py \
  --task So101Bench-Bin-v0 --episodes_jsonl tasks/custom_bin.jsonl \
  --policy_host localhost --policy_port 5556 \
  --action_horizon 16 --use_overhead_init false --headless \
  --record_dataset   # add to capture video for motion inspection (factor #1 above)
```
