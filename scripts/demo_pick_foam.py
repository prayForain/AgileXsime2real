#!/usr/bin/env python3
"""Run 1 automatic IK-planned grasp demo in PiperXPickFoam-v1 using SAPIEN's
native PinocchioModel IK (zero extra planner crashes, exact URDF match).

Saves:
  - debug/sim_pick_foam_storyboard.jpg (4 stages x [Global + Wrist camera])
  - debug/sim_pick_foam_demo.mp4       (full rollout video)
"""

import math
import sys
from pathlib import Path

import cv2
import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import sapien
from transforms3d.euler import euler2quat, quat2euler

SIM2REAL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SIM2REAL_ROOT))
import piperx_sim  # noqa: E402, F401

DEBUG_DIR = SIM2REAL_ROOT / "debug"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

GRIPPER_OPEN = 0.045   # 90 mm total width
GRIPPER_CLOSE = 0.010  # firm compliant grip on 45 mm foam block


def to_rgb_uint8(t):
    arr = t.cpu().numpy() if hasattr(t, "cpu") else np.asarray(t)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return arr


def solve_topdown_ik(pm, tcp_idx: int, xyz, yaw_rad: float, init_qpos: np.ndarray, qlimits: np.ndarray):
    """Solve top-down 6-DoF IK for tcp_link, grasping strictly across the 5cm narrow side
    of the 10x5x5 cm rectangular foam block (180-deg symmetric: 0 and pi)."""
    # Wrap yaw into [-pi/2, pi/2] (180-deg symmetry for rectangular block)
    base_yaw = ((yaw_rad + math.pi / 2) % math.pi) - math.pi / 2
    best_q = None
    best_cost = float("inf")

    for offset in (math.pi, 0.0):
        q_wxyz = euler2quat(math.pi, 0.0, base_yaw + offset, axes="sxyz")
        pose = sapien.Pose(p=np.asarray(xyz, dtype=np.float64), q=np.asarray(q_wxyz, dtype=np.float64))
        q_sol, ok, err = pm.compute_inverse_kinematics(
            tcp_idx,
            pose,
            initial_qpos=init_qpos,
            active_qmask=[1, 1, 1, 1, 1, 1, 0, 0],
            eps=1e-4,
            max_iterations=300,
        )
        if not ok:
            continue
        q6 = q_sol[:6]
        # Check within joint limits with 0.02 rad safety margin
        if np.all(q6 >= qlimits[:6, 0] + 0.02) and np.all(q6 <= qlimits[:6, 1] - 0.02):
            cost = float(np.linalg.norm(q6 - init_qpos[:6]))
            if cost < best_cost:
                best_cost = cost
                best_q = q6.copy()

    if best_q is None:
        raise RuntimeError(f"IK failed to find in-limit solution for xyz={xyz}, yaw={yaw_rad}")
    return best_q


def move_smooth(env, target_q6: np.ndarray, gripper_val: float, steps: int, frames_g, frames_w):
    """Cosine-eased joint interpolation from current qpos to target_q6."""
    cur_q6 = env.unwrapped.agent.robot.get_qpos()[0].cpu().numpy()[:6]
    obs, info = None, {}
    for k in range(1, steps + 1):
        alpha = 0.5 * (1.0 - math.cos(math.pi * k / steps))
        q_interp = (1.0 - alpha) * cur_q6 + alpha * target_q6
        act = np.concatenate([q_interp, [gripper_val]]).astype(np.float32)
        obs, _, _, _, info = env.step(act)
        frames_g.append(to_rgb_uint8(obs["sensor_data"]["global_camera"]["rgb"]))
        frames_w.append(to_rgb_uint8(obs["sensor_data"]["wrist_camera"]["rgb"]))
    # Settle 4 frames
    act_final = np.concatenate([target_q6, [gripper_val]]).astype(np.float32)
    for _ in range(4):
        obs, _, _, _, info = env.step(act_final)
        frames_g.append(to_rgb_uint8(obs["sensor_data"]["global_camera"]["rgb"]))
        frames_w.append(to_rgb_uint8(obs["sensor_data"]["wrist_camera"]["rgb"]))
    return obs, info


def hold_gripper(env, q6: np.ndarray, gripper_val: float, steps: int, frames_g, frames_w):
    act = np.concatenate([q6, [gripper_val]]).astype(np.float32)
    obs, info = None, {}
    for _ in range(steps):
        obs, _, _, _, info = env.step(act)
        frames_g.append(to_rgb_uint8(obs["sensor_data"]["global_camera"]["rgb"]))
        frames_w.append(to_rgb_uint8(obs["sensor_data"]["wrist_camera"]["rgb"]))
    return obs, info


def main():
    env = gym.make(
        "PiperXPickFoam-v1",
        obs_mode="rgbd",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        sim_backend="cpu",
    )
    obs, _ = env.reset(seed=42)
    robot = env.unwrapped.agent.robot
    pm = robot.create_pinocchio_model()
    tcp_idx = [l.name for l in robot.get_links()].index("tcp_link")
    qlimits = robot.get_qlimits()[0].cpu().numpy()

    foam_pose = env.unwrapped.foam.pose
    foam_p = foam_pose.p[0].cpu().numpy()
    foam_q = foam_pose.q[0].cpu().numpy().astype(np.float64)
    _, _, foam_yaw = quat2euler(foam_q, axes="sxyz")
    print(f"Black foam spawned at XYZ={np.round(foam_p, 4)}, yaw={math.degrees(foam_yaw):.1f} deg")

    frames_g = [to_rgb_uint8(obs["sensor_data"]["global_camera"]["rgb"])]
    frames_w = [to_rgb_uint8(obs["sensor_data"]["wrist_camera"]["rgb"])]
    key_idx = {"1. Start": 0}

    cur_q = robot.get_qpos()[0].cpu().numpy().astype(np.float64)

    # Stage 1: Pre-grasp directly above foam block
    q_pre = solve_topdown_ik(pm, tcp_idx, [foam_p[0], foam_p[1], foam_p[2] + 0.10], foam_yaw, cur_q, qlimits)
    print("Stage 1: Moving to Pre-Grasp...", np.round(np.degrees(q_pre), 1))
    obs, info = move_smooth(env, q_pre, GRIPPER_OPEN, steps=24, frames_g=frames_g, frames_w=frames_w)
    key_idx["2. Pre-Grasp"] = len(frames_g) - 1

    # Stage 2: Descend straight down onto foam block
    q_pre_full = np.concatenate([q_pre, [GRIPPER_OPEN, GRIPPER_OPEN]])
    q_grasp = solve_topdown_ik(pm, tcp_idx, [foam_p[0], foam_p[1], foam_p[2] + 0.002], foam_yaw, q_pre_full, qlimits)
    print("Stage 2: Descending to Grasp...", np.round(np.degrees(q_grasp), 1))
    obs, info = move_smooth(env, q_grasp, GRIPPER_OPEN, steps=20, frames_g=frames_g, frames_w=frames_w)

    # Stage 3: Close gripper firmly onto foam
    print("Stage 3: Closing Gripper...")
    obs, info = hold_gripper(env, q_grasp, GRIPPER_CLOSE, steps=18, frames_g=frames_g, frames_w=frames_w)
    key_idx["3. Grasped"] = len(frames_g) - 1

    # Stage 4: Lift 13 cm straight up
    q_grasp_full = np.concatenate([q_grasp, [GRIPPER_CLOSE, GRIPPER_CLOSE]])
    q_lift = solve_topdown_ik(pm, tcp_idx, [foam_p[0], foam_p[1], foam_p[2] + 0.13], foam_yaw, q_grasp_full, qlimits)
    print("Stage 4: Lifting Foam Block...", np.round(np.degrees(q_lift), 1))
    obs, info = move_smooth(env, q_lift, GRIPPER_CLOSE, steps=22, frames_g=frames_g, frames_w=frames_w)
    obs, info = hold_gripper(env, q_lift, GRIPPER_CLOSE, steps=10, frames_g=frames_g, frames_w=frames_w)
    key_idx["4. Lifted"] = len(frames_g) - 1

    success = bool(info["success"][0].item())
    is_grasped = bool(info["is_grasped"][0].item())
    foam_z = float(info["foam_z"][0].item())
    print(f"\nResult: success={success} | is_grasped={is_grasped} | foam_z={foam_z*100:.1f} cm | total_frames={len(frames_g)}")

    # Save 2x4 storyboard
    top_tiles, bot_tiles = [], []
    for label, idx in key_idx.items():
        g_img = frames_g[idx].copy()
        w_img = frames_w[idx].copy()
        cv2.putText(g_img, f"{label} (Global)", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2)
        cv2.putText(w_img, f"{label} (Wrist)", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2)
        top_tiles.append(g_img)
        bot_tiles.append(w_img)
    grid = np.vstack([np.hstack(top_tiles), np.hstack(bot_tiles)])
    sb_path = DEBUG_DIR / "sim_pick_foam_storyboard.jpg"
    cv2.imwrite(str(sb_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"Saved storyboard: {sb_path}")

    # Save side-by-side H.264 MP4 video (universally playable in browsers/IDEs)
    import shutil, subprocess
    import imageio_ffmpeg

    vid_raw = DEBUG_DIR / "sim_pick_foam_demo_raw.mp4"
    vid_path = DEBUG_DIR / "sim_pick_foam_demo.mp4"
    h, w, _ = frames_g[0].shape
    writer = cv2.VideoWriter(str(vid_raw), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (w * 2, h))
    for g, wr in zip(frames_g, frames_w):
        writer.write(cv2.cvtColor(np.hstack([g, wr]), cv2.COLOR_RGB2BGR))
    writer.release()
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(vid_raw), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(vid_path)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    vid_raw.unlink(missing_ok=True)
    print(f"Saved H.264 video: {vid_path}")

    env.close()


if __name__ == "__main__":
    main()
