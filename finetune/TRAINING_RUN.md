# GR00T N1.7 fine-tune — run reports

## Full run (20,000 steps) — 2026-07-07

Completed cleanly on the SO-101 sim dataset. Command as in the recipe with
`STEPS=20000 SAVE_FREQ=5000 BATCH_SIZE=8 WANDB_ENABLE=false`,
`OUTPUT_DIR=outputs/train/so101_bench_groot_n1.7`.

**Loss at each checkpoint** (relative actions, gripper excluded; backbone frozen):

| Checkpoint | Loss | grad-norm | LR | epoch |
|---|---|---|---|---|
| step 10 (start) | 1.114 | 0.76 | ~0 | 0.00 |
| 5,000 | 0.137 | 0.78 | 9.0e-5 | 0.21 |
| 10,000 | 0.098 | 0.63 | 5.6e-5 | 0.44 |
| 15,000 | 0.082 | 0.54 | 1.8e-5 | 0.67 |
| **20,000 (final)** | **0.065** | 0.56 | 1.7e-7 | 0.90 |

- Clean monotonic convergence **1.114 → 0.065 (~17×)**; min single-batch loss 0.047.
- Wall time **88.5 min** at ~3.9 steps/s; ~0.9 epoch of the 480-episode / 173K-frame set.
- GPU ~36 GB (batch 8); grad-norms stable ~0.5–0.8, cosine LR → ~0.
- Output: `outputs/train/so101_bench_groot_n1.7/checkpoints/{005000,010000,015000,020000,last}`,
  each a complete LeRobot `GrootPolicy` (~24 GB: model + optimizer/scheduler/RNG state).

> Log-parsing note: lerobot logs steps ≥1000 with a `K` suffix (`step:5K`, `step:20K`) and
> also block-buffers stdout — so grepping `step:5000` finds nothing mid-run. Use `PYTHONUNBUFFERED=1`
> and match `step:5K` for live parsing, or just read the checkpoint losses at the end.

### Checkpoint card — `checkpoints/020000` (= `last`)

The deliverable checkpoint. Config verified from
`checkpoints/020000/pretrained_model/config.json`:

| Field | Value |
|---|---|
| Policy | `groot` (LeRobot `GrootPolicy`), base `nvidia/GR00T-N1.7-3B` |
| Embodiment tag | `new_embodiment` |
| Cameras (`input_features`) | `observation.images.front` + `observation.images.overhead`, `3×480×640` (internal `image_size` 256×256) |
| State / action | `observation.state` 6-dim, `action` 6-dim (padded to `max_*_dim` 132) |
| Chunk / action steps | `chunk_size=16`, `n_action_steps=16` |
| Relative actions | `use_relative_actions=true`, `relative_exclude_joints=["gripper"]` |
| Trainable | `tune_projector` + `tune_diffusion_model` + `tune_vlln` = **true**; `tune_llm` / `tune_visual` = **false** (backbone frozen) → 1.62 B / 3.14 B params |
| Precision | `use_bf16=true`, `model_params_fp32=true`, `use_peft=false` (`lora_rank=0`) |
| Optimizer | AdamW `lr=1e-4`, betas (0.9, 0.999), wd 1e-5, `warmup_ratio=0.05` |
| Final train loss | **0.065** (step 20,000) |
| On-disk | ~24 GB (`model.safetensors` ~12.6 GB + preprocessor + optimizer/RNG state) |

> ⚠️ **Don't read run settings from `config.json`.** Its `max_steps:10000`, `batch_size:32`,
> `save_steps:1000`, `report_to:wandb`, `output_dir:./tmp/gr00t` are the policy-class
> **defaults**, not this run's actual values (20,000 steps, batch 8, save every 5,000,
> `WANDB_ENABLE=false`). Those live in the launch command / this report, not the saved config.

### Downstream eval

First digital-twin eval of this checkpoint (`So101Bench-Bin-v0`, 5-episode `custom_bin`,
served via `zmq_bridge_server.py`): **0/5 success — all `time_out`.** A recorded rollout
confirms the arm **moves and reaches the correct objects but misses the fine grasp**
(genuine geometric-execution gap, not a broken integration). Full analysis, per-episode
table, and the buffering gotcha that first hid the results: **[`EVAL_RUN.md`](./EVAL_RUN.md)**.

---

## Smoke run (10 steps) — 2026-07-06

A short, real end-to-end validation of the SO-101 Bench → GR00T N1.7 fine-tune pipeline
(`finetune/finetune_groot_n1.7.sh`) on `5hadytru/so101_bench_sim_1`. This was a 10-step
smoke run to prove the plumbing; scale `STEPS` for a real fine-tune (see the end).

Run date: 2026-07-06.

## Command

```bash
export PATH=~/Documents/medium_blog/grootn1.7/lerobot/.venv/bin:$PATH   # N1.7 lerobot
cd ~/Documents/medium_blog/so101_bench_groot_n1.7
STEPS=10 SAVE_FREQ=1000 WANDB_ENABLE=false BATCH_SIZE=8 \
  OUTPUT_DIR=<scratch>/groot_smoke_out \
  ./finetune/finetune_groot_n1.7.sh
```

Effective `lerobot-train` flags: `--policy.type=groot`
`--policy.base_model_path=nvidia/GR00T-N1.7-3B` `--policy.embodiment_tag=new_embodiment`
`--policy.chunk_size=16 --policy.n_action_steps=16` `--policy.use_relative_actions=true`
`--policy.relative_exclude_joints=["gripper"]` `--policy.use_bf16=true`
`--dataset.revision=main` `--env_eval_freq=0 --eval_steps=0`.

## Environment

| | |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell (~96 GB) |
| lerobot | grootn1.7 checkout N1.7 (`groot_n1_7`), `.venv` |
| torch / CUDA | 2.11.0+cu128 |
| transformers | 5.5.4 (Qwen3-VL) |
| accelerate | 1.14.0 |
| wandb | not installed → `WANDB_ENABLE=false` |

## Inputs downloaded

| Artifact | Size | Notes |
|---|---|---|
| dataset `5hadytru/so101_bench_sim_1` | 2.2 GB | 480 episodes, 173,612 frames, 30 fps, `so101_follower`, cameras `front` + `overhead` |
| model `nvidia/GR00T-N1.7-3B` | ~5.6 GB | full checkpoint (backbone weights + action head) |
| backbone `nvidia/Cosmos-Reason2-2B` | 28 KB | **gated** — only `config.json` needed (backbone weights come from the GR00T checkpoint) |

## Result

```
step:10  smpl:80  ep:0  epch:0.00  loss:1.114  grdn:0.761  lr:1.1e-06
         updt_s:0.346  data_s:0.981  smp/s:6  mem_gb:35.96
INFO ot_train.py:641 Checkpoint policy after step 10
INFO ot_train.py:721 End of training
```

| Metric | Value |
|---|---|
| Training loss (step 10) | **1.114** |
| Grad norm | 0.761 |
| LR (warmup) | 1.1e-6 |
| Trainable params | 1,620,515,968 (1.62 B) |
| Total params | 3,144,016,000 (3.14 B) — backbone frozen |
| GPU (process) | ~36 GB at batch 8 |
| Throughput | ~2.5 step/s (first step ~11 s compile), ~6 samples/s |
| Checkpoint | saved after step 10; exit code 0 |

Interpretation: `loss ≈ 1.11` after 10 warmup steps is just a "the graph runs and the
flow-matching MSE is being minimized" signal — not a quality number. Only the projector +
flow-matching action head + VL-LN train (1.62 B of 3.14 B); the Cosmos-Reason2/Qwen3-VL
backbone stays frozen.

## Issues resolved to get here

1. N1.5 vs N1.7 `lerobot` on PATH → run from the N1.7 checkout (preflight guard added).
2. CLI flags differ per build → N1.7 uses `--env_eval_freq`/`--eval_steps` + relative-action flags.
3. Dataset had no `codebase_version` tag → `--dataset.revision=main`.
4. `batch_size=64` OOM risk → default `BATCH_SIZE=8`.
5. Gated backbone `nvidia/Cosmos-Reason2-2B` (403) → accept the repo terms on HF.

See `../GROOT_N1.7_WORK.md` §4 for detail.

## Scale to a real fine-tune

```bash
export PATH=~/Documents/medium_blog/grootn1.7/lerobot/.venv/bin:$PATH
cd ~/Documents/medium_blog/so101_bench_groot_n1.7
pip install wandb        # optional; else keep WANDB_ENABLE=false
STEPS=20000 SAVE_FREQ=5000 BATCH_SIZE=8 \
  OUTPUT_DIR=outputs/train/so101_bench_groot_n1.7 \
  ./finetune/finetune_groot_n1.7.sh
```

~20k steps ≈ a couple hours at this throughput; checkpoints land under
`OUTPUT_DIR/checkpoints/`. Raise `BATCH_SIZE` (headroom exists), enable `PUSH_TO_HUB`/`REPO_ID`
to publish, and use `add_overhead_init.py` + the augmented dataset for the WM variant.
Evaluate the resulting LeRobot checkpoint with `lerobot-eval` / `lerobot-rollout` (not the
NVIDIA ZMQ server — see `README.md`).
