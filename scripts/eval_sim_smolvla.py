#!/usr/bin/env python3
"""Closed-loop evaluation of fine-tuned SmolVLA inside ManiSkill 3 (PiperXPickFoam-v1).

Connects to smolvla_policy_server.py over localhost TCP, runs N unseen episodes,
prints success rate, and saves an H.264 evaluation video.
"""

import argparse
import base64
import json
import socket
import struct
import subprocess
import sys
from pathlib import Path

import cv2
import gymnasium as gym
import imageio_ffmpeg
import mani_skill.envs  # noqa: F401
import numpy as np

SIM2REAL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SIM2REAL_ROOT))
import piperx_sim  # noqa: E402, F401

DEBUG_DIR = SIM2REAL_ROOT / "debug"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)


def recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Server disconnected")
        buf.extend(chunk)
    return bytes(buf)


def rpc(conn: socket.socket, payload: dict) -> dict:
    raw = json.dumps(payload).encode("utf-8")
    conn.sendall(struct.pack("!I", len(raw)) + raw)
    (length,) = struct.unpack("!I", recv_exact(conn, 4))
    return json.loads(recv_exact(conn, length).decode("utf-8"))


def to_rgb_uint8(t):
    arr = t.cpu().numpy() if hasattr(t, "cpu") else np.asarray(t)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def get_state_7d(robot) -> list[float]:
    q = robot.get_qpos()[0].cpu().numpy().astype(np.float32)
    return [float(q[0]), float(q[1]), float(q[2]), float(q[3]), float(q[4]), float(q[5]), float(q[6] + q[7])]


def main():
    parser = argparse.ArgumentParser(description="Evaluate SmolVLA closed-loop in ManiSkill 3")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=140)
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--task", type=str, default="Pick up the black foam block.")
    parser.add_argument("--domain-randomize", action="store_true")
    parser.add_argument(
        "--video-out",
        type=str,
        default=str(DEBUG_DIR / "sim_smolvla_eval.mp4"),
    )
    args = parser.parse_args()

    env = gym.make(
        "PiperXPickFoam-v1",
        obs_mode="rgbd",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        sim_backend="cpu",
        domain_randomize=args.domain_randomize,
    )

    successes = 0
    video_frames = []

    with socket.create_connection((args.host, args.port), timeout=60.0) as conn:
        for ep in range(args.num_episodes):
            obs, _ = env.reset(seed=args.seed + ep)
            robot = env.unwrapped.agent.robot
            rpc(conn, {"cmd": "reset"})

            ep_success = False
            max_z = 0.0
            for step in range(args.max_steps):
                cam1 = to_rgb_uint8(obs["sensor_data"]["global_camera"]["rgb"])
                cam2 = to_rgb_uint8(obs["sensor_data"]["wrist_camera"]["rgb"])
                state = get_state_7d(robot)

                resp = rpc(
                    conn,
                    {
                        "cmd": "step",
                        "camera1_b64": base64.b64encode(cam1.tobytes()).decode("ascii"),
                        "camera1_shape": list(cam1.shape),
                        "camera2_b64": base64.b64encode(cam2.tobytes()).decode("ascii"),
                        "camera2_shape": list(cam2.shape),
                        "state": state,
                        "task": args.task,
                    },
                )
                act7 = np.asarray(resp["action"], dtype=np.float32)
                # Convert total gripper width [0, 0.10] -> per-finger [0, 0.05] for ManiSkill
                sim_act = np.concatenate([act7[:6], [np.clip(act7[6] * 0.5, 0.0, 0.05)]]).astype(np.float32)
                obs, _, _, _, info = env.step(sim_act)

                foam_z = float(info["foam_z"][0].item())
                max_z = max(max_z, foam_z)
                if bool(info["success"][0].item()):
                    ep_success = True

                # Record first 3 episodes into the preview video
                if ep < 3:
                    pair = np.hstack([cam1, cam2])
                    tag = f"Ep {ep+1} | step={step+1:3d} | z={foam_z*100:4.1f}cm | {'OK' if ep_success else '...'}"
                    cv2.putText(pair, tag, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2)
                    video_frames.append(pair)

                if ep_success and step >= 100:
                    break

            successes += int(ep_success)
            print(f"Episode {ep+1:2d}/{args.num_episodes} (seed={args.seed+ep}): success={ep_success} | max_foam_z={max_z*100:.1f} cm", flush=True)

    env.close()
    rate = successes / max(1, args.num_episodes) * 100.0
    print(f"\n=== SmolVLA Sim Eval Result: {successes}/{args.num_episodes} ({rate:.1f}%) ===")

    if video_frames:
        vid_raw = Path(args.video_out).with_suffix(".raw.mp4")
        h, w, _ = video_frames[0].shape
        writer = cv2.VideoWriter(str(vid_raw), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (w, h))
        for fr in video_frames:
            writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        writer.release()
        subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-y",
                "-i",
                str(vid_raw),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                args.video_out,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        vid_raw.unlink(missing_ok=True)
        print(f"Saved H.264 eval video: {args.video_out}")


if __name__ == "__main__":
    main()
