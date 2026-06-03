# SO-101 Bench Isaac Lab Environment

This extension turns the Isaac Lab template into an SO-101 Bench environment for
language-conditioned tabletop manipulation with a GR00T-N1.6 fine-tune. It uses
an orange SO-101 arm, a constant plastic bin, one to four tabletop objects, wrist
and overhead RGB-D cameras, and the four benchmark task families from the paper:

- `So101Bench-Bin-v0`: place each object, or the single object, in the plastic bin.
- `So101Bench-Bin-SingleObject-v0`: bin task with exactly one randomly selected object slot active on the table.
- `So101Bench-Bin-Object1-v0` through `So101Bench-Bin-Object4-v0`: bin task with a specific object slot active.
- `So101Bench-NextTo-v0`: place one object next to another.
- `So101Bench-Between-v0`: place one object between two referents.
- `So101Bench-Move-v0`: move one object in a commanded direction.
- `So101Bench-Mixed-v0`: sample among all four families.

The environment uses local USD assets for the bedroom tabletop, plastic bin, and
tabletop objects.

## Required Local Asset Path

The bundled scan at
`source/so101_bench/so101_bench/assets/usd/room_scan.usdc` is used by default
when present. To use a different bedroom/tabletop USD, update
`BEDROOM_TABLETOP_USD` in
`source/so101_bench/so101_bench/tasks/direct/so101_bench/so101_bench_env_cfg.py`.

For the bundled scan, `collision_mesh/Plane_002` is treated as the collidable
tabletop mesh, while `table_visual` and `room` remain visual-only. The scan is
authored at real-world scale, with the tabletop origin centered on the tabletop
surface, so the environment loads it with identity scale and rotation.

## Install

Use the Python interpreter that has Isaac Lab installed:

```bash
python -m pip install -e source/so101_bench
```

If Isaac Lab is not on your shell `python`, use your Isaac Lab launcher:

```bash
/home/truman/IsaacLab/isaaclab.sh -p -m pip install -e source/so101_bench
```

## Smoke Tests

List registered tasks:

```bash
/home/truman/IsaacLab/isaaclab.sh -p scripts/list_envs.py
```

Run the environment with a zero-action debug agent:

```bash
/home/truman/IsaacLab/isaaclab.sh -p scripts/zero_agent.py --task So101Bench-Bin-v0 --enable_cameras
```

Inspect the first JSONL episode without stepping physics:

```bash
/home/truman/IsaacLab/isaaclab.sh -p scripts/groot_eval.py \
  --task So101Bench-Bin-v0 \
  --episodes_jsonl tasks/real_gr00t_seen_1obj_WM.jsonl \
  --inspect_initial_scene
```

## GR00T Remote Inference

Start your GR00T-N1.6 fine-tuned policy server separately, then run:

```bash
/home/truman/IsaacLab/isaaclab.sh -p scripts/groot_eval.py \
  --task So101Bench-Bin-v0 \
  --episodes_jsonl tasks/real_gr00t_seen_1obj_WM.jsonl \
  --policy_host localhost \
  --policy_port 5555
```

`--episodes_jsonl` is required by `scripts/groot_eval.py`. It drives the
episode objects and instruction from each JSONL row. Every object name is
validated against `OBJECT_SPLITS` in
`so101_bench/benchmark.py` before evaluation. Object labels map to local USD
filenames by replacing spaces with underscores, so `"green shoes"` selects
`assets/usd/objects/green_shoes.usdc`. Slots whose object USDs contain multiple
rigid bodies are spawned as `AssetBaseCfg`; single-rigid-body objects use
`RigidObjectCfg`.

By default the evaluator runs every validated JSONL row. Use
`--num_episodes 10` to evaluate only the first 10 rows.

Rows must provide `objects` and `instruction`. The instruction must be one of
the four benchmark forms:

```json
{"objects": ["grey wires"], "instruction": "Place each object in the plastic bin"}
{"objects": ["black glasses", "silver glasses", "yellow toy car", "cardboard box"], "instruction": "Place the yellow toy car next to the silver glasses."}
{"objects": ["black glasses", "silver glasses", "yellow toy car", "cardboard box"], "instruction": "Place the cardboard box between the black glasses and the yellow toy car."}
{"objects": ["black glasses", "silver glasses", "yellow toy car", "cardboard box"], "instruction": "Move the cardboard box forwards."}
```

For WM-conditioned checkpoints that expect the real-robot `overhead_init`
camera stream, enable the fixed settled-frame input. The evaluator holds the
initial robot pose for 1 second by default, then records `overhead_init` and
starts querying GR00T:

```bash
/home/truman/IsaacLab/isaaclab.sh -p scripts/groot_eval.py \
  --task So101Bench-Bin-v0 \
  --episodes_jsonl tasks/real_gr00t_seen_1obj_WM.jsonl \
  --policy_host localhost \
  --policy_port 5555 \
  --action_horizon 16 \
  --use_overhead_init true
```

By default, the script sends the current JSONL row instruction to the policy.
To override only the policy language, pass a matching fixed instruction:

```bash
/home/truman/IsaacLab/isaaclab.sh -p scripts/groot_eval.py \
  --task So101Bench-Bin-v0 \
  --episodes_jsonl tasks/real_gr00t_seen_1obj_WM.jsonl \
  --lang_instruction "Place each object in the plastic bin."
```

The sim camera names are `wrist` and `overhead`. The evaluator sends the wrist
camera as policy key `front` by default to match the SO100/SO101 real-robot
GR00T scripts. If your fine-tune expects different camera keys, pass a rename
map:

```bash
--rename_map '{"wrist":"ego","overhead":"external"}'
```

## MolmoAct2 Zero-Shot Inference

The released
[`allenai/MolmoAct2-SO100_101`](https://huggingface.co/allenai/MolmoAct2-SO100_101)
checkpoint can run as a zero-shot SO-101 policy. Start its HTTP server in a
separate Python environment, preferably on a GPU that has at least 16 GB free
for `bfloat16` inference:

```bash
pip install torch torchvision transformers accelerate safetensors pillow numpy \
  huggingface-hub einops requests fastapi uvicorn

python scripts/molmoact2_server.py \
  --device cuda:0 \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --port 8000
```

Then run the benchmark evaluator:

```bash
/home/truman/IsaacLab/isaaclab.sh -p scripts/molmoact2_eval.py \
  --task So101Bench-Bin-v0 \
  --episodes_jsonl tasks/test_each_seen_single.jsonl \
  --policy_host localhost \
  --policy_port 8000
```

The MolmoAct2 checkpoint expects two RGB images. The evaluator sends
`overhead,wrist` by default. The public SO-101 deployment notes that its
training data used two third-person views, so this is also worth comparing:

```bash
--policy_cameras overhead,overhead
```

`scripts/molmoact2_eval.py` applies the released model's absolute joint-pose
frame conversion and limits each commanded joint-target update to 15 degrees
by default. Use `--max_joint_step_deg 0` only when intentionally disabling
that clamp.

## LeRobot Dataset Replay

Replay a recorded LeRobot episode in the simulator by applying its saved
`action` stream to the SO-101 robot:

```bash
/home/truman/IsaacLab/isaaclab.sh -p scripts/so101_lerobot_replay.py \
  --task So101Bench-Bin-v0 \
  --episodes_jsonl tasks/test3.jsonl \
  --episode_layouts_jsonl tasks/layouts/test3_layouts_20260524_203554.jsonl \
  --repo_root data/lerobot/so101_bench_follower_teleop \
  --repo_id 5hadytru/so101_bench_sim_1 \
  --dataset_episode_index 0 \
  --benchmark_episode_index 0 \
  --real_time
```

`--dataset_episode_index` selects the LeRobot episode. `--benchmark_episode_index`
selects the JSONL/layout row used to reset the benchmark scene; it defaults to
the dataset episode index for sequential recordings. For skipped/cancelled
teleop rows, use `--benchmark_episode_indices 0,2,5`. Pass the layout JSONL
saved by teleop to replay the original object and bin poses exactly. During
replay, press `P` to pause, `N` to skip, or `Q` to quit.

For batched outcome collection, replay several episodes concurrently inside one
Isaac Lab process with `--num_envs`:

```bash
/home/truman/IsaacLab/isaaclab.sh -p scripts/so101_lerobot_collect_outcomes.py \
  --headless \
  --num_envs 4 \
  --frame_source none \
  --repo_root data/lerobot/so101_bench_sim_1_v3.0
```

Each Isaac environment receives its own recorded action stream and is refilled
independently when its episode finishes. `scripts/run_collect_outcomes_sharded.sh`
also accepts `NATIVE_ENVS=4` when process-level and native parallelism should be
combined.

Collected and rescored episode records include `final_diagnostics`, with the
condition-level geometry behind the final label. Re-evaluate selected saved
trajectories after changing a rule without launching Isaac Sim:

```bash
source ~/env_isaaclab_51/bin/activate
python scripts/so101_rescore_outcomes.py \
  --outcomes_dir data/lerobot/so101_bench_sim_1_v3.0/eval/sim_replay_outcomes_20260601_203813 \
  --episode_indices 6,32,70
```

## Teleop Object Placement Debugging

Add `--debug_object_placement` to the usual teleop command to save a
human-readable report and top-down SVGs for the selected layouts:

```bash
source ~/env_isaaclab_51/bin/activate
python scripts/so101_follower_teleop.py \
  --task So101Bench-Bin-v0 \
  --episodes_jsonl tasks/real_world_results_seen_spatial_instruction_following_6a.jsonl \
  --leader xbox \
  --no_record \
  --debug_object_placement
```

The script writes `summary.txt`, `summary.jsonl`, `index.html`, and one SVG per
episode in a sibling directory named
`<layouts-jsonl-stem>_object_placement_debug`. Provided layouts are rechecked
against the current spatial-layout rules.

The wrist camera defaults to a 640x480 render so the sim sends the same image
shape as the real-robot OpenCV command. To compare/tune policy frames:

```bash
/home/truman/IsaacLab/isaaclab.sh -p scripts/view_wrist_camera.py \
  --task So101Bench-Bin-v0 \
  --camera wrist \
  --steps 1 \
  --save_every 1
```

Useful wrist-camera tuning knobs live in
`source/so101_bench/so101_bench/tasks/direct/so101_bench/so101_bench_env_cfg.py`:

```python
INNOMAKER_WRIST_CAMERA_HORIZONTAL_FOV_DEG = 102.0
INNOMAKER_WRIST_CAMERA_POS = (-0.005, 0.060, -0.062)
INNOMAKER_WRIST_CAMERA_RPY_DEG = (-45.0, 0.0, 0.0)
```

## Paper-Derived Evaluation Logic

The simulator includes automatic checks for the measurable rules in the paper:

- maximum of three usable grasp attempts per target object, with automatic
  failure on the fourth eligible close cycle;
- plastic bin displaced by more than 1 inch;
- non-target object displaced by more than 0.5 inches;
- move-task boundary object displaced by more than 0.5 inches;
- all active-object containment for bin placement;
- closest-surface and no-grasped-object-contact checks for next-to placement;
- the between-task 1.5-inch COM-to-line rule plus centering and
  no-grasped-object-contact checks;
- directional move boundaries selected from the nearest laterally overlapping
  blocking object, with a 2-inch progress fallback when none exists, plus
  automatic trajectory-straightness failure, no-crossing, and
  no-grasped-object-contact checks.

An illustrated map of the non-bin task logic is available as an editable
[`SVG`](docs/non_bin_evaluation_logic.svg) and a ready-to-use
[`PNG`](docs/non_bin_evaluation_logic.png). The footprint and bounding-box
pipeline has a companion [`SVG`](docs/footprint_geometry.svg) and
[`PNG`](docs/footprint_geometry.png).

The initial-scene layout path is documented separately in a detailed editable
[`SVG`](docs/object_placement_computation.svg). It covers the shared
footprint-aware sampler, each task family's solvability filter, and exact JSONL
layout replay during environment reset.

Qualitative appendix labels such as semantic error, bad grasp strategy,
occlusion-induced grasp failure, and failed reorientation remain annotation
categories. They are preserved in `so101_bench/benchmark.py` but cannot be
reliably inferred from geometry alone.

## Files To Customize Next

- Bedroom/tabletop USD path: `BEDROOM_TABLETOP_USD` in `so101_bench_env_cfg.py`
- Tabletop collision subtree: `BEDROOM_TABLETOP_COLLISION_PRIM` in `so101_bench_env_cfg.py`
- Real object mesh registry: `OBJECT_SPLITS` in `so101_bench/benchmark.py`
  plus matching USD files in `source/so101_bench/so101_bench/assets/usd/objects`.
- Camera key mapping for your GR00T fine-tune: `--rename_map` in
  `scripts/groot_eval.py`
- Wrist camera match to your real setup: `INNOMAKER_WRIST_CAMERA_HORIZONTAL_FOV_DEG`,
  `INNOMAKER_WRIST_CAMERA_POS`, and `INNOMAKER_WRIST_CAMERA_RPY_DEG` in
  `source/so101_bench/so101_bench/tasks/direct/so101_bench/so101_bench_env_cfg.py`.

## Some commands

python gr00t/eval/run_gr00t_server.py   --model-path ~/workspace/so101_GR00T-N1.6-3B_WM_v6_55k/checkpoint-55000/   --embodiment-tag NEW_EMBODIMENT   --device cuda   --host 127.0.0.1   --port 5555

gsettings set org.gnome.mutter check-alive-timeout 0

/home/truman/IsaacLab/isaaclab.sh -p scripts/groot_eval.py   --task So101Bench-Bin-v0   --episodes_jsonl tasks/real_gr00t_seen_1obj_WM.jsonl   --policy_host localhost   --policy_port 5555   --action_horizon 16   --use_overhead_init true

/home/truman/IsaacLab/isaaclab.sh -p scripts/view_wrist_camera.py   --task So101Bench-Bin-v0   --display

/home/truman/IsaacLab/isaaclab.sh -p scripts/zero_agent.py --task So101Bench-Bin-v0 --enable_cameras --device cpu
