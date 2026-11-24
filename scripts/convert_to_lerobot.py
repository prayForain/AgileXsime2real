#!/usr/bin/env python3
"""Convert collected PiperXPickFoam .npz episodes into a standard LeRobotDataset.

Uses camera1 (global) and camera2 (wrist) keys matching lerobot/smolvla_base's
native camera input slots out-of-the-box.
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

SIM2REAL_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Convert raw .npz episodes to LeRobotDataset")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=str(SIM2REAL_ROOT / "data" / "raw_episodes"),
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default=str(SIM2REAL_ROOT / "data" / "piperx_pick_foam_lerobot"),
    )
    parser.add_argument("--repo-id", type=str, default="local/piperx_pick_foam")
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    raw_files = sorted(Path(args.raw_dir).glob("ep_*.npz"))
    if not raw_files:
        raise SystemExit(f"ERROR: No ep_*.npz files found in {args.raw_dir}")

    out_root = Path(args.out_root)
    if out_root.exists():
        shutil.rmtree(out_root)

    # Probe image shape from first episode
    sample = np.load(raw_files[0], allow_pickle=True)
    h, w, c = sample["global_rgb"].shape[1:]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper_m"],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper_m"],
        },
        "observation.images.camera1": {
            "dtype": "video",
            "shape": (h, w, c),
            "names": ["height", "width", "channels"],
        },
        "observation.images.camera2": {
            "dtype": "video",
            "shape": (h, w, c),
            "names": ["height", "width", "channels"],
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=features,
        root=out_root,
        robot_type="piper_x",
        use_videos=True,
    )

    total_frames = 0
    for idx, fpath in enumerate(raw_files):
        ep = np.load(fpath, allow_pickle=True)
        states = ep["state"].astype(np.float32)
        actions = ep["action"].astype(np.float32)
        g_imgs = ep["global_rgb"]
        w_imgs = ep["wrist_rgb"]
        task = str(ep["task"].item()) if ep["task"].ndim == 0 else str(ep["task"])

        T = len(states)
        for t in range(T):
            dataset.add_frame(
                {
                    "observation.state": states[t],
                    "action": actions[t],
                    "observation.images.camera1": g_imgs[t],
                    "observation.images.camera2": w_imgs[t],
                    "task": task,
                }
            )
        dataset.save_episode()
        total_frames += T
        if (idx + 1) % 20 == 0 or (idx + 1) == len(raw_files):
            print(f"  Converted [{idx + 1:3d}/{len(raw_files)}] episodes ({total_frames} frames)")

    dataset.finalize()
    print(f"\nLeRobotDataset saved to: {out_root}")
    print(f"Total episodes: {len(raw_files)} | Total frames: {total_frames}")


if __name__ == "__main__":
    main()
