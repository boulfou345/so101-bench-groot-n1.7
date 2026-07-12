# SO-101 Bench × GR00T-N1.7

**Fine-tune and evaluate GR00T-N1.7 on SO-101 Bench in an Isaac Lab digital twin — no robot required.**

This is the GR00T-N1.7 variant of [`5hadytru/so101_bench`](https://github.com/5hadytru/so101_bench):
a simulated twin of a language-conditioned SO-101 tabletop manipulation benchmark
(bin / next-to / between / move tasks over 56 household objects), retargeted from
GR00T-N1.6 to **N1.7** (Cosmos-Reason2 / Qwen3-VL backbone). See the upstream repo
for the benchmark design and the paper's real-robot findings.

- [Setup](#setup)
- [Fine-tuning](#fine-tuning)
- [Evaluation](#evaluation)
- [Workflow](#workflow)

---

## Setup

Two separate Python environments are required — their `transformers` pins conflict:

| Environment | Used for | Key stack |
|---|---|---|
| Isaac Lab (`~/IsaacLab/isaaclab.sh`) | Simulator + evaluator | Isaac Sim 5.1, Isaac Lab 5.1 |
| N1.7 LeRobot venv (`grootn1.7/lerobot/.venv`) | Fine-tuning + policy serving | LeRobot `groot_n1_7`, transformers 5.x |

**1. Install the benchmark extension into the Isaac Lab env** (must point at *this*
clone — an editable install from another clone will shadow it with stale modules):

```bash
~/IsaacLab/isaaclab.sh -p -m pip install -e source/so101_bench
```

**2. Download the USD assets** (~430 MB of room/bin/arm/object meshes; gitignored):

```bash
huggingface-cli download 5hadytru/so101_bench_assets so101_bench_usd_assets.tar.gz \
  --repo-type dataset --local-dir /tmp/so101_assets
tar -xzf /tmp/so101_assets/so101_bench_usd_assets.tar.gz \
  -C source/so101_bench/so101_bench/assets/
```

**3. Set up the N1.7 LeRobot env** (fine-tune + serving):

```bash
cd ../grootn1.7/lerobot && pip install -e '.[groot,training]'
```

Prerequisite: accept the gated
[`nvidia/Cosmos-Reason2-2B`](https://huggingface.co/nvidia/Cosmos-Reason2-2B)
backbone repo with your Hugging Face account, or policy loading dies with a 403.

**4. Smoke-test the sim:**

```bash
~/IsaacLab/isaaclab.sh -p scripts/list_envs.py
```

A containerized alternative (sim image + policy-server image, wired via compose)
lives under [`docker/`](docker/README.md).

---

## Fine-tuning

Entry point: [`finetune/finetune_groot_n1.7.sh`](finetune/finetune_groot_n1.7.sh)
(LeRobot `lerobot-train`, `GrootPolicy`). Run it inside the N1.7 LeRobot venv — a
preflight guard refuses to run on an N1.5 (Eagle-backbone) LeRobot build.

```bash
BATCH_SIZE=8 ./finetune/finetune_groot_n1.7.sh
```

### Reference configuration

These defaults produced a converged run on a single RTX PRO 6000 (~36 GB VRAM,
~90 min): loss 1.114 → 0.137 (5k) → 0.098 (10k) → 0.082 (15k) → **0.065 (20k)**.

**Model & data**

| Parameter | Value |
|---|---|
| Base model | `nvidia/GR00T-N1.7-3B` (Cosmos-Reason2 / Qwen3-VL backbone) |
| Dataset | `5hadytru/so101_bench_sim_1` (480 teleop episodes, 173,612 frames, 30 fps, `front` + `overhead`) |
| Embodiment tag | `new_embodiment` |
| Image augmentation | `dataset.image_transforms.enable=true` |

**Trainable components** — the VLM is frozen; only the action pathway trains
(no LoRA, `lora_rank=0`):

| Component | Trained |
|---|:---:|
| LLM backbone (`tune_llm`) | ❌ |
| Vision encoder (`tune_visual`) | ❌ |
| Projector (`tune_projector`) | ✅ |
| Diffusion action head (`tune_diffusion_model`) | ✅ |

**Action space**

| Parameter | Value |
|---|---|
| `chunk_size` / `n_action_steps` | 16 (keep in sync with eval `--action_horizon`) |
| `use_relative_actions` | `true` (joint deltas) |
| `relative_exclude_joints` | `["gripper"]` (gripper absolute) |

**Optimization**

| Parameter | Value |
|---|---|
| Optimizer | AdamW, lr `1e-4`, weight decay `1e-5`, betas (0.9, 0.999), grad clip 1.0 |
| LR schedule | cosine, 500 warmup steps (5%) |
| Precision | bf16 |
| Batch size / steps | 8 / 20,000 (reference recipe's 64 needs multi-GPU) |
| Checkpoint every | 5,000 steps |
| Seed | 42 |

All knobs are env-var overridable (`STEPS`, `BATCH_SIZE`, `DATASET`, `CHUNK_SIZE`,
`RELATIVE_ACTIONS`, `WANDB_ENABLE`, `PUSH_TO_HUB`/`REPO_ID`, …). Multi-GPU goes
through `finetune/finetune_groot_n1.7_multigpu.sh` (`NUM_GPUS`, FSDP via
`ACCELERATE_ARGS`). For the world-model-conditioned variant, first augment the
dataset with `finetune/add_overhead_init.py`, then train on the augmented set and
evaluate with `--use_overhead_init true`.

Full recipe notes, VRAM guidance, and gotchas: [`finetune/README.md`](finetune/README.md);
a worked training log: [`finetune/TRAINING_RUN.md`](finetune/TRAINING_RUN.md).

---

## Evaluation

`lerobot-train` writes a **LeRobot-format** checkpoint, which NVIDIA's ZMQ server
cannot load. [`finetune/zmq_bridge_server.py`](finetune/zmq_bridge_server.py)
serves it over the same ZMQ/msgpack wire contract the evaluator speaks.

### One command (GUI)

```bash
./scripts/launch_groot_n17_gui.sh --num_episodes 5
```

Auto-starts the bridge on `:5556` (if not already up), waits for the checkpoint to
load, then launches `groot_eval.py` with the Isaac Sim window, recording to a
timestamped `data/lerobot/groot_n17_gui_<timestamp>` dataset. Overrides:
`N17_CKPT`, `BRIDGE_PORT`, `N17_VENV_PY`, `REPO_ROOT`; extra flags pass through to
`groot_eval.py`. During a run: `P` snapshots all cameras, `N` skips to the next
episode.

### Manual (headless sweeps)

```bash
# 1. bridge (N1.7 lerobot venv)
PYTHONUNBUFFERED=1 python finetune/zmq_bridge_server.py \
  --model-path outputs/train/so101_bench_groot_n1.7/checkpoints/last/pretrained_model \
  --host 0.0.0.0 --port 5556 --action-horizon 16

# 2. evaluator (Isaac Lab)
PYTHONUNBUFFERED=1 ~/IsaacLab/isaaclab.sh -p scripts/groot_eval.py \
  --task So101Bench-Bin-v0 \
  --episodes_jsonl tasks/real_gr00t_WM_combined.jsonl \
  --policy_host localhost --policy_port 5556 \
  --action_horizon 16 --use_overhead_init false \
  --record_dataset --repo_root data/lerobot/groot_n17_eval \
  --headless --num_episodes 5
```

Key flags:

- `--episodes_jsonl` *(required)* — task file; one row per episode (`objects` + `instruction`).
- `--num_episodes N` — keep small interactively: without it, layouts for **all**
  rows are pre-sampled before the window opens (the 281-row file can OOM-kill the process).
- `--use_overhead_init` — `true` only for WM-conditioned checkpoints.
- `--episode_layouts_jsonl` — replay the exact object poses a previous run saved
  under `tasks/layouts/`.
- `--record_dataset` — save wrist/overhead video + states/actions as a LeRobot
  dataset for failure analysis. Never share one `--repo_root` between two live runs.

Scoring is automatic (grasp-attempt cap, bin containment, next-to radius, between
center-line, move boundaries); each episode ends with `success=`, an end reason,
and a failure taxonomy (`failed_grasp`, `wrong_object`, …). A worked evaluation
report of the 20k checkpoint: [`finetune/EVAL_RUN.md`](finetune/EVAL_RUN.md).

### Pitfalls

- **`NotImplementedError: …lerobot_dataset is missing`** — the Isaac env's
  editable `so101_bench` points at a different clone. Reinstall (Setup step 1).
- **`FileNotFoundError: … assets/usd/SO-ARM101-USD.usd`** — assets missing in the
  installed clone (Setup step 2, or symlink an existing download).
- **HuggingFace 404 at recorder init** — the `--repo_root` holds a half-written
  dataset (a second eval collided with a running one). Use a fresh directory.
- **Results vanish when piping to a file** — Isaac's hard exit drops buffered
  stdout; always run with `PYTHONUNBUFFERED=1` (the launch scripts do).

---

## Workflow

```
┌─────────────────────────┐         ZMQ (msgpack)          ┌──────────────────────────┐
│  Isaac Lab digital twin │ ─── observations ────────────► │  GR00T-N1.7 policy       │
│  (so101_bench env)      │      video.front/overhead,     │  LeRobot checkpoint      │
│  scene + physics +      │      joint state, instruction  │  behind zmq_bridge_server│
│  automatic scoring      │ ◄─── action chunks ─────────── │  on :5556                │
└─────────────────────────┘      16 joint targets/query    └──────────────────────────┘
```

1. **Scene construction** — each episode is a JSONL row (`objects` + `instruction`).
   The env spawns scanned USD meshes and samples a physically feasible layout from
   per-object footprint polygons; layouts are saved for exact replay.
2. **Fine-tune** — `lerobot-train` on the SO-101 sim teleop dataset: frozen VLM,
   trained projector + diffusion action head, relative actions, chunk 16.
3. **Serve** — the ZMQ bridge loads the LeRobot checkpoint plus its saved pre/post
   processors and answers the generation-agnostic `ping`/`reset`/`get_action`
   protocol — the sim client is identical for N1.6 and N1.7.
4. **Roll out** — `groot_eval.py` steps the sim at 30 Hz, queries the policy every
   16 steps (~0.53 s), executes the returned chunk, and records everything as a
   LeRobot dataset.
5. **Score** — the paper's measurable rules run automatically and every episode
   gets a machine-graded verdict plus condition-level geometry diagnostics.

The wire contract is model-agnostic, so swapping policy generations (or models —
see `scripts/molmoact2_eval.py`) changes only the serving side, never the sim.

---

## Citation

```bibtex
@misc{hickok2025so101bench,
  title  = {SO-101 Bench: Measuring the Gap Between Semantic and Geometric Competence in Vision-Language-Action Models},
  author = {Hickok, Truman},
  year   = {2025}
}
```
