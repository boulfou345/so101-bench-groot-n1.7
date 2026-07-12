#!/usr/bin/env python
"""Add a working-memory ``observation.images.overhead_init`` modality to a LeRobot dataset.

SO-101 Bench's WM conditioning feeds the policy the *settled overhead frame captured at
the start of the episode*, held constant for the whole episode, as an extra video input
alongside the live front/overhead cameras. The public teleop dataset
(``5hadytru/so101_bench_sim_1``) ships only ``observation.images.front`` and
``observation.images.overhead``, so a WM fine-tune needs this extra column synthesized
first.

This script reads a source LeRobot dataset and writes a NEW dataset that is identical
except for an added ``observation.images.overhead_init`` feature, where every frame of an
episode carries that episode's **first overhead frame** (the closest offline analog of
the settled start-of-episode frame). Point the fine-tune at the new dataset.

Source ``5hadytru/so101_bench_sim_1`` has cameras ``observation.images.front`` and
``observation.images.overhead`` (480 episodes, ``so101_follower``); this adds a third.

Run it in the LeRobot (grootn1.7) checkout, in the ``[groot]`` environment:

    python finetune/add_overhead_init.py \
        --src-repo-id 5hadytru/so101_bench_sim_1 \
        --dst-repo-id $HF_USER/so101_bench_sim_1_wm \
        --limit-episodes 1        # smoke-test on one episode first

Then fine-tune with DATASET=$HF_USER/so101_bench_sim_1_wm, and at eval keep
``groot_eval.py --use_overhead_init true --overhead_init_key overhead_init``.

NOTE: the LeRobotDataset write API (create/add_frame/save_episode/finalize),
``episode_data_index``, and the per-frame ``task`` key are version-sensitive. Validate
on ``--limit-episodes 1`` before running the whole set, and adjust to your installed
LeRobot version if a field name differs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset

DEFAULT_OVERHEAD_KEY = "observation.images.overhead"
DEFAULT_INIT_KEY = "observation.images.overhead_init"
# Bookkeeping columns that LeRobotDataset.create()/add_frame() manage themselves.
_MANAGED = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


def to_hwc_uint8(image: torch.Tensor | np.ndarray) -> np.ndarray:
    """Decoded LeRobot images are CHW float in [0, 1]; add_frame wants HWC uint8."""
    array = image.detach().cpu().numpy() if isinstance(image, torch.Tensor) else np.asarray(image)
    if array.ndim == 3 and array.shape[0] in (1, 3, 4):  # CHW -> HWC
        array = np.transpose(array, (1, 2, 0))
    if array.shape[-1] == 1:  # grayscale -> RGB
        array = np.repeat(array, 3, axis=-1)
    if not np.issubdtype(array.dtype, np.uint8):
        if float(array.max(initial=0.0)) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def episode_bounds(src: LeRobotDataset, episode_index: int) -> tuple[int, int]:
    """Return [from, to) global frame indices for an episode."""
    idx = src.episode_data_index
    return int(idx["from"][episode_index]), int(idx["to"][episode_index])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-repo-id", default="5hadytru/so101_bench_sim_1")
    ap.add_argument("--src-root", default=None, help="Local dir for the source dataset (else HF cache).")
    ap.add_argument("--dst-repo-id", required=True, help="Repo id for the augmented dataset.")
    ap.add_argument("--dst-root", default=None, help="Local dir for the new dataset (else HF cache).")
    ap.add_argument("--overhead-key", default=DEFAULT_OVERHEAD_KEY)
    ap.add_argument("--init-key", default=DEFAULT_INIT_KEY)
    ap.add_argument("--limit-episodes", type=int, default=None, help="Process only the first N episodes.")
    ap.add_argument("--no-videos", action="store_true", help="Store images instead of MP4 video.")
    args = ap.parse_args()

    src = LeRobotDataset(args.src_repo_id, root=args.src_root)
    if args.overhead_key not in src.features:
        raise SystemExit(
            f"Source has no {args.overhead_key!r}. Available image keys: "
            f"{[k for k in src.features if k.startswith('observation.images.')]}"
        )
    if args.init_key in src.features:
        raise SystemExit(f"Source already has {args.init_key!r}; nothing to do.")

    # Clone the feature spec, dropping managed bookkeeping cols, and add the WM modality
    # as a copy of the overhead camera's spec.
    features = {k: dict(v) for k, v in src.features.items() if k not in _MANAGED}
    features[args.init_key] = dict(features[args.overhead_key])

    dst = LeRobotDataset.create(
        repo_id=args.dst_repo_id,
        fps=src.fps,
        features=features,
        root=args.dst_root,
        robot_type=src.meta.robot_type,
        use_videos=not args.no_videos,
    )

    num_episodes = src.num_episodes if args.limit_episodes is None else min(args.limit_episodes, src.num_episodes)
    print(f"Augmenting {num_episodes}/{src.num_episodes} episodes: +{args.init_key}")

    for episode_index in range(num_episodes):
        start, end = episode_bounds(src, episode_index)
        # The settled start-of-episode overhead frame, reused for every frame below.
        init_frame = to_hwc_uint8(src[start][args.overhead_key])

        for global_index in range(start, end):
            item = src[global_index]
            frame: dict = {args.init_key: init_frame}
            for key in features:
                if key == args.init_key:
                    continue
                value = item[key]
                if key.startswith("observation.images."):
                    value = to_hwc_uint8(value)
                frame[key] = value
            # add_frame requires the language/task string.
            frame["task"] = item["task"] if isinstance(item.get("task"), str) else str(item.get("task", ""))
            dst.add_frame(frame)

        dst.save_episode()
        print(f"  episode {episode_index}: {end - start} frames")

    dst.finalize()
    root = Path(args.dst_root) if args.dst_root else "(HF cache)"
    print(f"Done. Wrote {args.dst_repo_id} -> {root}")
    print("Fine-tune with DATASET=" + args.dst_repo_id + " and eval with --use_overhead_init true.")


if __name__ == "__main__":
    main()
