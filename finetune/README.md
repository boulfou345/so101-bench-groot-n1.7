# Fine-tuning GR00T N1.7 for SO-101 Bench

This directory holds a runnable recipe to **fine-tune GR00T N1.7 on the SO-101 Bench
teleop dataset** using LeRobot's `lerobot-train` (the `GrootPolicy` reimplementation),
producing a checkpoint you can then evaluate in the digital twin.

- Script: [`finetune_groot_n1.7.sh`](./finetune_groot_n1.7.sh)
- Base model: [`nvidia/GR00T-N1.7-3B`](https://huggingface.co/nvidia/GR00T-N1.7-3B)
- Dataset: [`5hadytru/so101_bench_sim_1`](https://huggingface.co/datasets/5hadytru/so101_bench_sim_1)
  (SO-101 sim teleop — 480 episodes, `so101_follower`)

> [!IMPORTANT]
> **Checkpoint format ≠ NVIDIA's N1.5 ZMQ server.** `lerobot-train` writes a
> **LeRobot-format** GR00T checkpoint (`GrootPolicy`). NVIDIA's Isaac-GR00T ZMQ server
> (`gr00t/eval/run_gr00t_server.py`) loads **Isaac-GR00T-format** checkpoints and **cannot**
> load this one — and GR00T **N1.7 has no such server anyway** (that flow was N1.5). Two
> supported ways to run this checkpoint (see [Evaluation](#evaluation)): **(A)** drive the
> Isaac Lab benchmark (`scripts/groot_eval.py`, a custom ZMQ client) through the included
> [`zmq_bridge_server.py`](./zmq_bridge_server.py), which serves the LeRobot `GrootPolicy` over
> the client's wire contract; or **(B)** LeRobot-native `lerobot-rollout` / `lerobot-eval`.

---

## 1. Environment

Fine-tuning runs in the **N1.7 LeRobot (`grootn1.7`) checkout**, not in this Isaac Lab
repo — `lerobot-train` and the N1.7 `groot` policy live there. Install it into a fresh
env:

```bash
# in the grootn1.7/lerobot checkout
pip install -e ".[groot,training]"      # or: uv sync --extra groot --extra training
hf auth login                           # dataset + (optional) checkpoint push
wandb login                             # optional; set WANDB_ENABLE=false to skip
```

> [!IMPORTANT]
> **Accept the gated backbone repo.** GR00T N1.7 loads its Cosmos-Reason2 / Qwen3-VL
> backbone + processor from **[`nvidia/Cosmos-Reason2-2B`](https://huggingface.co/nvidia/Cosmos-Reason2-2B)**,
> a **gated** repo. Without access, training dies at "Creating policy" with
> `GatedRepoError: 403 Forbidden` on `Cosmos-Reason2-2B/config.json` (even though
> `nvidia/GR00T-N1.7-3B` itself downloads fine). Open the model page with your training
> account, click **"Agree and access repository"**, then verify:
> ```bash
> python -c "from huggingface_hub import hf_hub_download as d; print(d('nvidia/Cosmos-Reason2-2B','config.json'))"
> ```
> `model_info(...)` succeeding is **not** enough — it returns metadata for gated repos;
> only a successful file download confirms access.

> [!WARNING]
> **Use the N1.7 checkout — not an older N1.5 `lerobot`.** GR00T has two incompatible
> generations in LeRobot: N1.5 (Eagle backbone, module `groot_n1`, base
> `nvidia/GR00T-N1.5-3B`) and N1.7 (Cosmos-Reason2 / Qwen3-VL, module `groot_n1_7`, base
> `nvidia/GR00T-N1.7-3B`). If your `lerobot-train` resolves to an N1.5 install, training
> `nvidia/GR00T-N1.7-3B` fails at model load. The script preflight-checks for the
> `groot_n1_7` module and stops with instructions if it's missing. Verify with:
> ```bash
> python -c "import lerobot.policies.groot.groot_n1_7 as m; print(m.__file__)"
> ```
> N1.7 needs the Qwen3-VL–capable `transformers` the `groot` extra pins (`>=5.4,<5.6`) —
> a **different** env from the `gr00t-server` image (`transformers==4.57.3` for NVIDIA's
> own N1.7 server) and from any N1.5 install. Keep them separate.

## 2. Data

The dataset is pulled from the Hub by `--dataset.repo_id` on first run (cached under
`~/.cache/huggingface`). To stage it explicitly / offline:

```bash
huggingface-cli download 5hadytru/so101_bench_sim_1 \
  --repo-type dataset --local-dir ~/datasets/so101_bench_sim_1
# then pass --dataset.root=~/datasets/so101_bench_sim_1 (add it to the script)
```

This dataset has no LeRobot `codebase_version` git tag, so LeRobot's default revision
lookup raises `RevisionNotFoundError`. The script pins `--dataset.revision=main` (override
via `DATASET_REVISION`) to read the data directly from the `main` branch.

Dataset shape (LeRobot v2): `observation.images.front`, `observation.images.overhead`,
`observation.state`, `action`, `robot_type: so101_follower`, 30 fps, 480 episodes. The
GR00T processor maps `observation.images.*` → the model's `video.*` modalities and the
per-episode task string → the language annotation; `observation.state`/`action` are
padded to 132 dims and normalized from the dataset statistics. (Note the wrist camera is
already named `front` here, matching GR00T's real-robot camera convention.)

## 3. Run

```bash
# from this repo; the script just invokes lerobot-train (must be on PATH)
./finetune/finetune_groot_n1.7.sh
```

Override any knob via the environment:

```bash
STEPS=30000 BATCH_SIZE=32 OUTPUT_DIR=outputs/train/so101_n17_run2 \
  ./finetune/finetune_groot_n1.7.sh

# push the result to the Hub
PUSH_TO_HUB=true REPO_ID=$HF_USER/so101_bench_GR00T_N1.7 \
  ./finetune/finetune_groot_n1.7.sh

# predict joint deltas (only on a lerobot build that supports it — see note below)
RELATIVE_ACTIONS=true ./finetune/finetune_groot_n1.7.sh
```

### Key hyperparameters (and why)

| Flag | Value | Rationale |
|---|---|---|
| `--policy.embodiment_tag` | `new_embodiment` | SO-101 uses GR00T's generic new-embodiment head (not `libero_sim`). |
| `--policy.chunk_size` / `n_action_steps` | `16` | Matches `groot_eval.py --action_horizon 16` (the WM replan cadence). |
| `--policy.use_bf16` | `true` | bf16 training for the 3B model. |
| `--dataset.image_transforms.enable` | `true` | Photometric aug helps real↔sim transfer. |
| `--batch_size` | `8` | Single-GPU-safe default. The SO-101 reference recipe uses 64; raise it if VRAM allows (below). |
| `--steps` / `--save_freq` | `20000` / `5000` | ~4 checkpoints; raise `steps` for a fuller run. |
| `--eval_freq` | `0` | No sim env in this training loop — periodic eval disabled. |

By default only the projector + diffusion head + VL-LN train (`tune_projector` /
`tune_diffusion_model` / `tune_vlln` on; `tune_llm` / `tune_visual` off) — the backbone
stays frozen.

> [!NOTE]
> **Relative actions are opt-in.** The official SO-101 `groot.mdx` recipe predicts joint
> *deltas* (`--policy.use_relative_actions=true`, gripper excluded), but not every
> installed `lerobot` build exposes those flags — passing them to a build that doesn't
> support them fails with `unrecognized arguments`. The script therefore defaults to
> **absolute actions** (which the SO-101 Bench eval already consumes) and only adds the
> relative-action flags when you set `RELATIVE_ACTIONS=true`. Enable it only on a build
> that supports them (e.g. the grootn1.7 checkout — `pip install -e ".[groot,training]"`
> from that repo). Likewise this script uses `--eval_freq=0`, not the older
> `--env_eval_freq` / `--eval_steps`.

## 4. Working-memory (`overhead_init`) conditioning

The SO-101 Bench **WM** setup conditions the policy on the settled overhead frame
captured at episode start (`overhead_init`) — an *extra* video input on top of live
front + overhead. The base recipe here trains the **standard two-camera** setup
(`front` + `overhead`) because `so101_bench_sim_1` ships only those two image columns.

To reproduce the WM baseline, augment the dataset with a third video modality
`observation.images.overhead_init` — a static per-episode copy of the settled overhead
frame — before training, so GrootPolicy learns to attend to it (and
`groot_eval.py --use_overhead_init true` then supplies it at inference).

Use the included helper [`add_overhead_init.py`](./add_overhead_init.py), which writes a
new dataset with that column (each frame carries its episode's first overhead frame):

```bash
# in the grootn1.7/lerobot checkout (groot env)
python finetune/add_overhead_init.py \
  --src-repo-id 5hadytru/so101_bench_sim_1 \
  --dst-repo-id $HF_USER/so101_bench_sim_1_wm \
  --limit-episodes 1        # smoke-test one episode first, then drop the flag

# then fine-tune on the augmented set
DATASET=$HF_USER/so101_bench_sim_1_wm ./finetune/finetune_groot_n1.7.sh
```

Without this step, train on the 2-camera set and run eval with
`--use_overhead_init false` so the observation shape matches what the checkpoint saw.
(The LeRobot dataset write API is version-sensitive — validate on one episode first;
see the script's docstring.)

## 5. Compute / VRAM

The backbone is ~3B params. The script defaults to `BATCH_SIZE=8` to fit a single GPU;
the SO-101 reference recipe's `batch_size=64` wants a large-VRAM GPU or, more
realistically, **multiple GPUs**. Options:

- Keep/reduce `BATCH_SIZE` (8, or lower to 4) on a single 24–48 GB GPU; raise it as VRAM allows.
- Multi-GPU via `accelerate` — use the included
  [`finetune_groot_n1.7_multigpu.sh`](./finetune_groot_n1.7_multigpu.sh)
  (`NUM_GPUS=4 BATCH_SIZE=16 ./finetune/finetune_groot_n1.7_multigpu.sh`; pass
  `ACCELERATE_ARGS="--config_file fsdp.yaml"` for FSDP). Remember `--batch_size` is
  **per-process**, so effective batch = `BATCH_SIZE * NUM_GPUS`. See
  `docs/source/multi_gpu_training.mdx` in the LeRobot checkout.
- Optionally set `--policy.lora_rank>0` for parameter-efficient fine-tuning (LoRA
  fields exist on `GrootConfig`).

## 6. Output layout

`lerobot-train` writes to `--output_dir`:

```
outputs/train/so101_bench_groot_n1.7/
  checkpoints/
    <step>/pretrained_model/    # config.json + model weights (LeRobot GrootPolicy)
    last/                       # symlink to newest
  ...                           # train logs / wandb run
```

Point evaluation/rollout at `checkpoints/<step>/pretrained_model` (or the pushed Hub
repo id).

## Evaluation

Two ways to run the trained checkpoint — pick by what you want to drive.

### A) Isaac Lab digital-twin benchmark (ZMQ bridge)

`scripts/groot_eval.py` is a **custom ZMQ client** (`PolicyClient` → `ping`/`reset`/`get_action`),
not `lerobot-rollout`. GR00T **N1.7 has no** NVIDIA `run_gr00t_server.py` flow (that was N1.5), and
this LeRobot checkpoint can't load in it anyway. So to drive the Isaac Lab sim with this checkpoint,
serve it through [`zmq_bridge_server.py`](./zmq_bridge_server.py): it loads the LeRobot `GrootPolicy`
(+ its saved pre/post processors) and speaks the exact wire contract the client expects.

```bash
# terminal 1 — in the grootn1.7/lerobot venv (same env used to fine-tune)
pip install pyzmq msgpack        # once, if missing
python finetune/zmq_bridge_server.py \
  --model-path outputs/train/so101_bench_groot_n1.7/checkpoints/last/pretrained_model \
  --host 0.0.0.0 --port 5555 --action-horizon 16
# sanity-check the checkpoint loads + infers, without ZMQ/sim:
#   python finetune/zmq_bridge_server.py --model-path <ckpt> --self-test

# terminal 2 — in the Isaac Lab env, point the eval at the bridge
./isaaclab.sh -p scripts/groot_eval.py \
  --task So101Bench-Bin-v0 --episodes_jsonl tasks/custom_bin.jsonl \
  --policy_host localhost --policy_port 5555 \
  --action_horizon 16 --use_overhead_init false --headless
```

Notes:
- Keep `--action-horizon` (bridge) == `--action_horizon` (eval).
- This checkpoint is the **2-camera** (front + overhead) variant — run with `--use_overhead_init false`
  so the observation shape matches what it was trained on. (For the WM variant, retrain with the
  augmented dataset from `add_overhead_init.py`; the bridge auto-detects the model's image features.)
- The bridge sends state in LeRobot `.pos` units, which is exactly what the client already sends
  (`sim_radians_to_raw_degrees` is an alias for `sim_radians_to_lerobot_positions`), so no unit
  conversion is applied. Relative-action decode is handled by the checkpoint's own postprocessor.

### B) LeRobot-native rollout (`lerobot-rollout`)

For hardware or a LeRobot-driven loop, use LeRobot's own inference path directly:

```bash
# in the grootn1.7/lerobot checkout
export MODEL=outputs/train/so101_bench_groot_n1.7/checkpoints/last/pretrained_model

lerobot-rollout \
  --strategy.type=base \
  --policy.path=$MODEL \
  --policy.base_model_path=nvidia/GR00T-N1.7-3B \
  --policy.n_action_steps=8 \
  --task="place each object in the plastic bin" \
  --device=cuda
```

See `docs/source/groot.mdx` (LeRobot) for the full rollout/eval reference and the
LIBERO benchmark commands.
