#!/usr/bin/env python3
"""Lightweight local TCP inference server for fine-tuned SmolVLA (runs in `lerobot` env).

Bridges the Python 3.12 `lerobot` environment with the Python 3.11 `piperx_vla`
ManiSkill 3 evaluation client using Python stdlib sockets + length-prefixed JSON/raw bytes.
"""

import argparse
import base64
import json
import socket
import struct
from pathlib import Path

import numpy as np
import torch
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Client disconnected")
        buf.extend(chunk)
    return bytes(buf)


def recv_msg(conn: socket.socket) -> dict:
    (length,) = struct.unpack("!I", recv_exact(conn, 4))
    return json.loads(recv_exact(conn, length).decode("utf-8"))


def send_msg(conn: socket.socket, payload: dict) -> None:
    raw = json.dumps(payload).encode("utf-8")
    conn.sendall(struct.pack("!I", len(raw)) + raw)


def decode_rgb(b64_str: str, shape: list[int]) -> torch.Tensor:
    arr = np.frombuffer(base64.b64decode(b64_str), dtype=np.uint8).reshape(shape).copy()
    # Convert HWC uint8 -> CHW float32 in [0, 1]
    return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0


def main():
    sim2real_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Local SmolVLA inference server")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(sim2real_root / "outputs" / "smolvla_pick_foam_aligned" / "checkpoints" / "005000" / "pretrained_model"),
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise SystemExit(f"ERROR: Checkpoint directory not found: {ckpt}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Loading SmolVLAPolicy from {ckpt} onto {device} ...", flush=True)
    policy = SmolVLAPolicy.from_pretrained(str(ckpt)).to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(ckpt),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    print(f"SmolVLA ready! Listening on {args.host}:{args.port}", flush=True)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((args.host, args.port))
        srv.listen(4)
        while True:
            conn, addr = srv.accept()
            print(f"Client connected: {addr}", flush=True)
            with conn:
                try:
                    while True:
                        req = recv_msg(conn)
                        cmd = req.get("cmd")
                        if cmd == "reset":
                            policy.reset()
                            send_msg(conn, {"ok": True})
                        elif cmd == "step":
                            cam1 = decode_rgb(req["camera1_b64"], req["camera1_shape"])
                            cam2 = decode_rgb(req["camera2_b64"], req["camera2_shape"])
                            state = torch.tensor(req["state"], dtype=torch.float32)
                            obs = {
                                "observation.state": state,
                                "observation.images.camera1": cam1,
                                "observation.images.camera2": cam2,
                                "task": req["task"],
                            }
                            with torch.no_grad():
                                batch = preprocessor(obs)
                                act = policy.select_action(batch)
                                act = postprocessor(act)
                            act_list = act.squeeze(0).detach().cpu().float().tolist()
                            send_msg(conn, {"ok": True, "action": act_list})
                        elif cmd == "shutdown":
                            send_msg(conn, {"ok": True})
                            print("Shutdown requested. Exiting.", flush=True)
                            return
                        else:
                            send_msg(conn, {"ok": False, "error": f"Unknown cmd: {cmd}"})
                except ConnectionError:
                    print(f"Client disconnected: {addr}", flush=True)


if __name__ == "__main__":
    main()
