#!/usr/bin/env python
"""ZMQ bridge server: serve a LeRobot GR00T N1.7 checkpoint over the SO-101 Bench wire.

Why this exists
---------------
`finetune/finetune_groot_n1.7.sh` trains GR00T **N1.7** with LeRobot's `lerobot-train`,
producing a **LeRobot-format** `GrootPolicy` checkpoint. GR00T N1.7 (unlike N1.5) has **no**
NVIDIA `run_gr00t_server.py` / `inference_service.py` ZMQ flow — it is served through LeRobot's
own CLIs. But the Isaac Lab benchmark in this repo (`scripts/groot_eval.py`) is a *custom* ZMQ
client (`PolicyClient` in `source/so101_bench/so101_bench/utils/groot.py`) that speaks NVIDIA's
`ping` / `reset` / `get_action` msgpack protocol.

This bridge closes that gap: it loads the LeRobot `GrootPolicy` (+ its saved pre/post
processors) and answers the exact wire contract `groot_eval.py` expects, so the existing
Isaac Lab eval and Docker sim run unchanged against a LeRobot-trained N1.7 checkpoint.

Wire contract (mirrors PolicyClient / GR00TRemotePolicy):
  * ZeroMQ REQ/REP, msgpack with numpy arrays `np.save`-encoded (see MsgSerializer below).
  * observation in (nested, two leading [1,1,...] batch/time axes):
      video.<cam>            uint8  [1,1,H,W,3]   (cam == policy image feature name)
      state.single_arm       float32[1,1,5]       LeRobot .pos units
      state.gripper          float32[1,1,1]       LeRobot .pos units
      language.annotation.human.task_description   [[str]]
  * action out: {"single_arm": [T,5], "gripper": [T,1]}, T = action_horizon.

State units: the client sends `sim_radians_to_raw_degrees`, which in this repo is an alias for
`sim_radians_to_lerobot_positions` (LeRobot .pos units) — the same units the checkpoint's
`observation.state` normalization was fit on, so no unit conversion is needed here.

Run inside the grootn1.7 LeRobot venv (same env used to fine-tune):
    pip install pyzmq msgpack          # if missing
    python finetune/zmq_bridge_server.py \
        --model-path outputs/train/so101_bench_groot_n1.7/checkpoints/last/pretrained_model \
        --host 0.0.0.0 --port 5555 --action-horizon 16

Validate without ZMQ/sim (loads the policy and runs one synthetic get_action):
    python finetune/zmq_bridge_server.py --model-path <ckpt> --self-test
"""

from __future__ import annotations

import argparse
import io
import traceback
from typing import Any

import msgpack
import numpy as np
import torch


# --------------------------------------------------------------------------------------
# msgpack serialization — byte-for-byte compatible with utils/groot.py MsgSerializer.
# --------------------------------------------------------------------------------------
class MsgSerializer:
    @staticmethod
    def to_bytes(data: Any) -> bytes:
        return msgpack.packb(data, default=MsgSerializer.encode_custom_classes)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        return msgpack.unpackb(data, object_hook=MsgSerializer.decode_custom_classes, raw=False)

    @staticmethod
    def decode_custom_classes(obj):
        if isinstance(obj, dict) and "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj

    @staticmethod
    def encode_custom_classes(obj):
        if isinstance(obj, np.ndarray):
            output = io.BytesIO()
            np.save(output, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": output.getvalue()}
        return obj


# --------------------------------------------------------------------------------------
# Policy wrapper
# --------------------------------------------------------------------------------------
class LeRobotGrootBridge:
    """Loads a LeRobot GrootPolicy checkpoint and turns a wire observation into an action chunk."""

    def __init__(self, model_path: str, device: str = "cuda", action_horizon: int = 16):
        from lerobot.policies.groot.modeling_groot import GrootPolicy
        from lerobot.policies import make_pre_post_processors
        from lerobot.utils.constants import ACTION  # noqa: F401  (imported for clarity/parity)

        print(f"[bridge] loading GrootPolicy from {model_path} ...", flush=True)
        self.policy = GrootPolicy.from_pretrained(model_path)
        self.policy.config.device = device
        self.policy.to(device)
        self.policy.eval()
        self.device = device
        self.action_horizon = action_horizon

        # Load the SAME linked pre/post processors saved with the checkpoint. The preprocessor's
        # pack step caches the raw state that the postprocessor's relative->absolute decode reads,
        # so we must call preprocessor(obs) then postprocessor(chunk) on these exact instances.
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=model_path,
            preprocessor_overrides={"device_processor": {"device": device}},
        )

        # Which image features the policy actually consumes (e.g. front, overhead).
        self.image_features = [
            k.split("observation.images.")[-1]
            for k in self.policy.config.input_features
            if k.startswith("observation.images.")
        ]
        self.action_dim = self.policy.config.output_features["action"].shape[0]
        print(
            f"[bridge] ready: images={self.image_features} action_dim={self.action_dim} "
            f"relative={getattr(self.policy.config, 'use_relative_actions', False)} device={device}",
            flush=True,
        )

    def reset(self):
        self.policy.reset()

    @staticmethod
    def _unwrap_scalar_str(value: Any) -> str:
        # language.annotation.human.task_description arrives double-wrapped: [[str]].
        while isinstance(value, (list, tuple)) and value:
            value = value[0]
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return str(value)

    def _obs_to_batch(self, obs: dict) -> dict:
        # --- state: single_arm (5) + gripper (1) -> observation.state [1, 6] ---
        state_block = obs["state"]
        single_arm = np.asarray(state_block["single_arm"], dtype=np.float32).reshape(-1)
        gripper = np.asarray(state_block["gripper"], dtype=np.float32).reshape(-1)
        state = np.concatenate([single_arm, gripper]).astype(np.float32)
        batch: dict[str, Any] = {
            "observation.state": torch.from_numpy(state).unsqueeze(0).to(self.device),
        }

        # --- images: video.<cam> uint8 [1,1,H,W,3] -> observation.images.<cam> [1,3,H,W] float ---
        video = obs.get("video", {})
        for cam in self.image_features:
            if cam not in video:
                raise KeyError(
                    f"observation is missing video.{cam!r}; got video keys {list(video.keys())}"
                )
            img = np.asarray(video[cam])
            img = img.reshape(img.shape[-3], img.shape[-2], img.shape[-1])  # -> [H, W, 3]
            t = torch.from_numpy(np.ascontiguousarray(img)).to(self.device)
            t = t.permute(2, 0, 1).unsqueeze(0).to(torch.float32) / 255.0  # [1, 3, H, W]
            batch[f"observation.images.{cam}"] = t

        # --- language -> task ---
        lang = obs.get("language", {}).get("annotation.human.task_description", "")
        batch["task"] = [self._unwrap_scalar_str(lang)]
        return batch

    @torch.no_grad()
    def get_action(self, obs: dict) -> dict:
        batch = self._obs_to_batch(obs)
        processed = self.preprocessor(batch)              # caches raw state for relative decode
        chunk = self.policy.predict_action_chunk(processed)  # [1, T, action_dim] (relative deltas)
        actions = self.postprocessor(chunk)               # -> absolute actions using cached state
        actions = actions.detach().to("cpu", torch.float32).numpy()
        if actions.ndim == 3:
            actions = actions[0]                           # [T, action_dim]
        actions = actions[: self.action_horizon]
        single_arm = np.ascontiguousarray(actions[:, :5], dtype=np.float32)
        gripper = np.ascontiguousarray(actions[:, 5:6], dtype=np.float32)
        return {"single_arm": single_arm, "gripper": gripper}


# --------------------------------------------------------------------------------------
# ZMQ REP loop
# --------------------------------------------------------------------------------------
def serve(bridge: LeRobotGrootBridge, host: str, port: int):
    import zmq

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://{host}:{port}")
    print(f"[bridge] serving on tcp://{host}:{port} — Ctrl-C to stop", flush=True)
    try:
        while True:
            request = MsgSerializer.from_bytes(sock.recv())
            endpoint = request.get("endpoint")
            try:
                if endpoint == "ping":
                    response: Any = "pong"
                elif endpoint == "reset":
                    bridge.reset()
                    response = {}
                elif endpoint == "get_action":
                    observation = request["data"]["observation"]
                    response = [bridge.get_action(observation), {}]
                else:
                    response = {"error": f"unknown endpoint {endpoint!r}"}
            except Exception as exc:  # report to client instead of dropping the socket
                traceback.print_exc()
                response = {"error": f"{type(exc).__name__}: {exc}"}
            sock.send(MsgSerializer.to_bytes(response))
    except KeyboardInterrupt:
        print("\n[bridge] shutting down", flush=True)
    finally:
        sock.close()
        ctx.term()


def _self_test(bridge: LeRobotGrootBridge):
    """Load-and-infer smoke test with a synthetic observation matching the wire contract."""
    print("[self-test] building synthetic observation ...", flush=True)
    obs = {
        "video": {
            cam: np.zeros((1, 1, 480, 640, 3), dtype=np.uint8) for cam in bridge.image_features
        },
        "state": {
            "single_arm": np.zeros((1, 1, 5), dtype=np.float32),
            "gripper": np.zeros((1, 1, 1), dtype=np.float32),
        },
        "language": {
            "annotation.human.task_description": [["Place each object in the plastic bin."]]
        },
    }
    action = bridge.get_action(obs)
    sa, gr = action["single_arm"], action["gripper"]
    print(f"[self-test] single_arm {sa.shape} {sa.dtype}  gripper {gr.shape} {gr.dtype}", flush=True)
    assert sa.shape == (bridge.action_horizon, 5), sa.shape
    assert gr.shape == (bridge.action_horizon, 1), gr.shape
    # round-trip through msgpack to prove the wire encoding works
    blob = MsgSerializer.to_bytes([action, {}])
    back = MsgSerializer.from_bytes(blob)
    assert np.allclose(back[0]["single_arm"], sa)
    print(f"[self-test] OK — msgpack round-trip {len(blob)} bytes; sample arm[0]={sa[0].tolist()}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", required=True, help="path to the LeRobot checkpoint pretrained_model dir")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--action-horizon", type=int, default=16,
                    help="steps returned per get_action; keep in sync with groot_eval.py --action_horizon")
    ap.add_argument("--self-test", action="store_true", help="load the policy, run one synthetic get_action, exit")
    args = ap.parse_args()

    bridge = LeRobotGrootBridge(args.model_path, device=args.device, action_horizon=args.action_horizon)
    if args.self_test:
        _self_test(bridge)
        return
    serve(bridge, args.host, args.port)


if __name__ == "__main__":
    main()
