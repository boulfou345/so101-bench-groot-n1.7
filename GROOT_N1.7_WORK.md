# SO-101 Bench → GR00T N1.7: work log

This directory (`so101_bench_groot_n1.7`) is a **GR00T N1.7 variant of SO-101 Bench**. It
was cloned from the original `so101_bench` (tracked code + git history, no heavy untracked
data) and modified to (a) evaluate GR00T **N1.7** in the Isaac Lab digital twin via the
existing remote-server pattern, and (b) fine-tune GR00T N1.7 on the SO-101 sim dataset with
LeRobot's `lerobot-train`.

The original `so101_bench` was left **untouched**; all changes live here.

---

## 1. Context

`so101_bench` drives an Isaac Lab SO-101 sim and queries a **remote GR00T policy server**
over ZeroMQ + msgpack (`scripts/groot_eval.py` → `GR00TRemotePolicy`/`PolicyClient` in
`source/so101_bench/so101_bench/utils/groot.py`). It was wired for GR00T **N1.6** (Eagle3-VL
backbone) via a working-memory (`overhead_init`) fine-tune.

Goal here: move to **N1.7** (Cosmos-Reason2 / Qwen3-VL backbone) while keeping the ZMQ
remote-server pattern and the WM conditioning.

**Key structural fact:** the ZMQ/msgpack wire contract (obs `video.*` / `state.single_arm` /
`state.gripper` / `annotation.human.task_description`; action `single_arm` / `gripper`) is a
GR00T-generation-agnostic convention that the Isaac-GR00T server maps from the checkpoint's
modality config. So the SO-101 client code is essentially unchanged N1.6 → N1.7; the real work
is the **server image**, a few **config values**, **docs**, and the **fine-tune recipe**.

---

## 2. Eval-tooling changes (N1.6 → N1.7)

| File | Change |
|---|---|
| `docker/gr00t-server/Dockerfile` | N1.7 server image: `transformers==4.51.3` (Eagle3-VL) → `4.57.3` (Qwen3-VL / Cosmos-Reason2); header, example run, `--embodiment-tag NEW_EMBODIMENT`; kept the cu128 Blackwell torch pin |
| `docker/compose.yaml` | N1.7 model path/env, `--embodiment-tag NEW_EMBODIMENT` on the server |
| `scripts/groot_eval.py` | `--action_horizon` help explains N1.7's 40-chunk vs the safe `16` default |
| `README.md` | Reframed GR00T-inference section to N1.7; added the fine-tune-first prerequisite pointing at `finetune/`; noted the wire contract is unchanged N1.6→N1.7 |
| `docker/README.md` | Table row, model path, run command, and the "what's baked in" pins rationale → N1.7 |

Not changed (generation-agnostic / hardware-only): `utils/groot.py` wire contract,
`SO101JointMapper`, `utils/lerobot_calibration.py`, the Isaac Lab env/tasks.

**Left as historical N1.6 (facts, not stale tooling):** the paper's real-world study numbers
in `README.md`, the `real_sim_1` dataset provenance lines, and `SETUP_FIXES.md` (a debugging
journal of the actual N1.6 bringup).

---

## 3. Fine-tune recipe (`finetune/`)

Pipeline chosen: **LeRobot `lerobot-train` (`GrootPolicy`)**, on **`5hadytru/so101_bench_sim_1`**
(480 episodes, `so101_follower`, cameras `observation.images.front` + `.overhead`).

| File | Purpose |
|---|---|
| `finetune/finetune_groot_n1.7.sh` | Parameterized `lerobot-train` command (adapted from `groot.mdx`): `new_embodiment`, `chunk_size=16`, relative actions (gripper excluded), bf16, `BATCH_SIZE=8` default. Env-var overridable; preflight-checks for the N1.7 build. |
| `finetune/finetune_groot_n1.7_multigpu.sh` | `accelerate` multi-GPU wrapper (sets `LERO_LAUNCH`); `NUM_GPUS`, `ACCELERATE_ARGS` (FSDP). |
| `finetune/add_overhead_init.py` | Synthesizes the WM `observation.images.overhead_init` column (episode's first overhead frame, held constant) into a new dataset — needed to reproduce the WM baseline. |
| `finetune/zmq_bridge_server.py` | **ZMQ bridge** so the Isaac Lab benchmark can drive the LeRobot checkpoint. Loads `GrootPolicy` + its saved pre/post processors, speaks the client's `ping`/`reset`/`get_action` msgpack contract, maps obs (`video.*`/`state.single_arm`+`gripper`/`annotation…`) → LeRobot batch, and returns `single_arm`+`gripper` chunks. `--self-test` loads+infers without ZMQ/sim. |
| `finetune/README.md` | Full recipe: env, data, hyperparameters, WM flow, VRAM/multi-GPU, output layout, eval (both the ZMQ-bridge and `lerobot-rollout` paths). |

**Why the bridge (and not NVIDIA's ZMQ server):** GR00T N1.7 has **no** `run_gr00t_server.py`
inference flow — that was N1.5. N1.7 is served via LeRobot's CLIs, and `lerobot-train` writes a
**LeRobot-format** checkpoint the NVIDIA server can't load. But `scripts/groot_eval.py` is a *custom*
ZMQ client, so the bridge re-serves the LeRobot policy over that same wire contract. Verified
end-to-end against the repo's own `PolicyClient` (ping/reset/get_action → 16-step `.pos` chunks).
Key non-obvious facts: state units already match (`sim_radians_to_raw_degrees` is an alias for
`sim_radians_to_lerobot_positions`); relative-action decode is done by the checkpoint's postprocessor,
which reads the reference state cached by the preprocessor's pack step — so the bridge must call the
linked pre/post processors in order and can't use `GrootPolicy.select_action` (it rejects relative
policies).

### Run it
```bash
# activate the N1.7 lerobot env (see §4), then:
BATCH_SIZE=8 ./finetune/finetune_groot_n1.7.sh
# WM variant: first `python finetune/add_overhead_init.py --dst-repo-id $HF_USER/..._wm`
```

---

## 4. Gotchas hit while getting it running (and the fixes)

These are the non-obvious things that each broke a run, in order:

1. **Two `lerobot` installs with different CLIs.** The default `~/lerobot` on PATH is GR00T
   **N1.5** (Eagle backbone, module `groot_n1`, base `nvidia/GR00T-N1.5-3B`) and cannot
   fine-tune N1.7. The N1.7 implementation is the separate **`grootn1.7/lerobot`** checkout
   (module `groot_n1_7`, Qwen3-VL). → Added a **preflight guard** in the script that requires
   the `groot_n1_7` module and stops with instructions otherwise.

2. **CLI flags differ between the two builds.**
   - N1.5 build: has `--eval_freq`, **no** relative-action flags.
   - N1.7 build (the correct target, matches `groot.mdx`): uses `--env_eval_freq` + `--eval_steps`,
     **and supports** `--policy.use_relative_actions` / `--policy.relative_exclude_joints`.
   → The script targets the **N1.7** build: `--env_eval_freq=0 --eval_steps=0`, relative actions
   on by default (`RELATIVE_ACTIONS=false` to disable).

3. **Dataset has no `codebase_version` git tag** → `RevisionNotFoundError`. → The script pins
   `--dataset.revision=main` (override via `DATASET_REVISION`).

4. **`nvidia/GR00T-N1.7-3B` on a 3B model OOMs at `batch_size=64`.** → Default lowered to
   `BATCH_SIZE=8` (single-GPU-safe; the reference recipe's 64 needs large VRAM / multi-GPU).

5. **`wandb` not installed** in the N1.7 venv → run with `WANDB_ENABLE=false` (or `pip install wandb`).

6. **Gated backbone repo.** GR00T N1.7 loads its Cosmos-Reason2 / Qwen3-VL backbone +
   processor from **`nvidia/Cosmos-Reason2-2B`**, which is **gated**. Training dies at
   "Creating policy" with `GatedRepoError: 403` unless the training HF account has accepted
   the repo terms (`nvidia/GR00T-N1.7-3B` itself is not gated and downloads fine). Note
   `model_info()` returns metadata for gated repos, so it looks accessible — only a
   successful `hf_hub_download(...)` confirms real file access. → Accept the gate at
   https://huggingface.co/nvidia/Cosmos-Reason2-2B .

### Env knobs
`DATASET`, `DATASET_REVISION`, `BASE_MODEL`, `EMBODIMENT_TAG`, `CHUNK_SIZE`, `BATCH_SIZE`,
`STEPS`, `SAVE_FREQ`, `SEED`, `OUTPUT_DIR`, `JOB_NAME`, `WANDB_ENABLE`, `PUSH_TO_HUB`/`REPO_ID`,
`RELATIVE_ACTIONS`, `SKIP_VERSION_CHECK`, `LERO_LAUNCH`; multi-GPU adds `NUM_GPUS`, `ACCELERATE_ARGS`.

---

## 5. Environment

- **N1.7 fine-tune / eval-with-lerobot:** the `grootn1.7/lerobot` checkout, `[groot,training]`
  extras, `transformers>=5.4,<5.6`. A prepared venv already exists at
  `grootn1.7/lerobot/.venv` (torch 2.11+cu128, transformers 5.5.4, accelerate 1.14 — `wandb`
  missing). Put its `bin/` on PATH so `python` + `lerobot-train` resolve to N1.7.
- **NVIDIA ZMQ server (Isaac-GR00T-format checkpoints):** separate image,
  `transformers==4.57.3` — see `docker/gr00t-server/Dockerfile`.
- These three environments (N1.7 lerobot / N1.7 server / N1.5 lerobot) are mutually
  incompatible on `transformers` and must stay separate.

Hardware here: NVIDIA RTX PRO 6000 Blackwell (~96 GB).

---

## 6. Important caveats

- **Checkpoint format:** `lerobot-train` writes a **LeRobot-format** `GrootPolicy` checkpoint.
  NVIDIA's Isaac-GR00T ZMQ server (used by `groot_eval.py`) loads **Isaac-GR00T-format**
  checkpoints and **cannot** load it directly — evaluate a lerobot-trained checkpoint with
  LeRobot's `lerobot-eval` / `lerobot-rollout`. Twin-based eval of a lerobot checkpoint would
  need a LeRobot-served policy behind the sim (separate task).
- **WM `overhead_init`:** `so101_bench_sim_1` ships only `front` + `overhead`. The base recipe
  trains those two; the WM baseline needs `add_overhead_init.py` first (and eval with
  `--use_overhead_init true`), else eval with `--use_overhead_init false`.
- **Relative vs absolute actions:** default relative (matches `groot.mdx` SO-101); set
  `RELATIVE_ACTIONS=false` for absolute if that better matches your deployment/eval path.

---

## 7. Status

- ✅ Eval tooling retargeted to N1.7 (server image, compose, docs).
- ✅ Fine-tune recipe authored + all three scripts syntax-validated.
- ✅ N1.7 env confirmed working (`groot_n1_7` imports, CUDA available).
- ✅ Training command accepted by the N1.7 `lerobot-train` (arg-parsing passes after the flag
  fixes above).
- ✅ Smoke run (10 steps) validated the pipeline end-to-end after accepting the
  `nvidia/Cosmos-Reason2-2B` gate.
- ✅ **Full run (20,000 steps) completed** (2026-07-07, ~88.5 min, batch 8, ~36 GB GPU).
  Loss converged **1.114 → 0.065 (~17×)**: 5k=0.137, 10k=0.098, 15k=0.082, 20k=0.065.
  4 checkpoints under `outputs/train/so101_bench_groot_n1.7/checkpoints/` (~24 GB each,
  complete LeRobot `GrootPolicy`). See `finetune/TRAINING_RUN.md`.
- ▶ Next: evaluate the checkpoint with LeRobot (`lerobot-eval` / `lerobot-rollout`) — it's a
  LeRobot-format checkpoint, not loadable by the NVIDIA ZMQ server (§6).
- ⛔ Prerequisite for real eval numbers: fine-tune to convergence, then evaluate (via LeRobot
  for a lerobot checkpoint, or the ZMQ server for an Isaac-GR00T-format one).
