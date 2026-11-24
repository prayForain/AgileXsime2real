#!/usr/bin/env python3
"""Batch automatic demonstration collector in ManiSkill 3 (PiperXPickFoam-v1).

Collects N successful domain-randomized episodes of PiperX grasping the
10cm x 5cm x 5cm matte black foam block across its 5cm narrow side.

State & Action convention (1:1 aligned with real PiperX pyAgxArm SDK!):
  - state[0:6]  : current joint angles J1..J6 in radians
  - state[6]    : current total gripper opening width in meters [0.0, 0.10]
  - action[0:6] : target joint angles J1..J6 in radians
  - action[6]   : target total gripper opening width in meters [0.0, 0.10]
"""

import argparse
import math
import sys
import time
from pathlib import Path

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import sapien
from transforms3d.euler import euler2quat, quat2euler

SIM2REAL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SIM2REAL_ROOT))
import piperx_sim  # noqa: E402, F401

GRIPPER_OPEN_FINGER = 0.045   # per finger -> 0.090 m total width
GRIPPER_CLOSE_FINGER = 0.010  # per finger -> 0.020 m target pinch (firm on 5cm foam)
TASK_TEXT = "Pick up the black foam block."


def to_rgb_uint8(t):
    arr = t.cpu().numpy() if hasattr(t, "cpu") else np.asarray(t)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return arr


def get_state_7d(robot) -> np.ndarray:
    q = robot.get_qpos()[0].cpu().numpy().astype(np.float32)
    # Total opening width = finger1 + finger2 (matches pyAgxArm move_gripper_m / value in meters)
    gripper_width = float(q[6] + q[7])
    return np.concatenate([q[:6], [gripper_width]]).astype(np.float32)


def solve_topdown_ik(pm, tcp_idx: int, xyz, yaw_rad: float, init_qpos: np.ndarray, qlimits: np.ndarray):
    base_yaw = ((yaw_rad + math.pi / 2) % math.pi) - math.pi / 2
    best_q = None
    best_cost = float("inf")

    for offset in (math.pi, 0.0):
        q_wxyz = euler2quat(math.pi, 0.0, base_yaw + offset, axes="sxyz")
        pose = sapien.Pose(p=np.asarray(xyz, dtype=np.float64), q=np.asarray(q_wxyz, dtype=np.float64))
        q_sol, ok, _ = pm.compute_inverse_kinematics(
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
        if np.all(q6 >= qlimits[:6, 0] + 0.02) and np.all(q6 <= qlimits[:6, 1] - 0.02):
            cost = float(np.linalg.norm(q6 - init_qpos[:6]))
            if cost < best_cost:
                best_cost = cost
                best_q = q6.copy()
    return best_q


def collect_one_episode(env, pm, tcp_idx: int, qlimits: np.ndarray, seed: int, rng: np.random.Generator):
    obs, _ = env.reset(seed=seed)
    robot = env.unwrapped.agent.robot

    foam_pose = env.unwrapped.foam.pose
    foam_p = foam_pose.p[0].cpu().numpy()
    foam_q = foam_pose.q[0].cpu().numpy().astype(np.float64)
    _, _, foam_yaw = quat2euler(foam_q, axes="sxyz")

    cur_q = robot.get_qpos()[0].cpu().numpy().astype(np.float64)

    # Slight randomized pre-grasp / lift heights for trajectory diversity
    pre_dz = float(rng.uniform(0.085, 0.115))
    lift_dz = float(rng.uniform(0.115, 0.145))

    q_pre = solve_topdown_ik(pm, tcp_idx, [foam_p[0], foam_p[1], foam_p[2] + pre_dz], foam_yaw, cur_q, qlimits)
    if q_pre is None:
        return None
    q_grasp = solve_topdown_ik(
        pm, tcp_idx, [foam_p[0], foam_p[1], foam_p[2] + 0.002], foam_yaw,
        np.concatenate([q_pre, [GRIPPER_OPEN_FINGER, GRIPPER_OPEN_FINGER]]), qlimits,
    )
    if q_grasp is None:
        return None
    q_lift = solve_topdown_ik(
        pm, tcp_idx, [foam_p[0], foam_p[1], foam_p[2] + lift_dz], foam_yaw,
        np.concatenate([q_grasp, [GRIPPER_CLOSE_FINGER, GRIPPER_CLOSE_FINGER]]), qlimits,
    )
    if q_lift is None:
        return None

    states, actions, g_imgs, w_imgs = [], [], [], []
    info = {}

    def record_step(target_q6: np.ndarray, finger_val: float):
        nonlocal obs, info
        states.append(get_state_7d(robot))
        g_imgs.append(to_rgb_uint8(obs["sensor_data"]["global_camera"]["rgb"]))
        w_imgs.append(to_rgb_uint8(obs["sensor_data"]["wrist_camera"]["rgb"]))
        # Store action gripper as total width (2 * finger_val) for 1:1 real PiperX match
        actions.append(np.concatenate([target_q6, [2.0 * finger_val]]).astype(np.float32))
        sim_act = np.concatenate([target_q6, [finger_val]]).astype(np.float32)
        obs, _, _, _, info = env.step(sim_act)

    def move_smooth(target_q6: np.ndarray, finger_val: float, steps: int):
        start_q6 = robot.get_qpos()[0].cpu().numpy()[:6]
        for k in range(1, steps + 1):
            alpha = 0.5 * (1.0 - math.cos(math.pi * k / steps))
            q_i = (1.0 - alpha) * start_q6 + alpha * target_q6
            record_step(q_i, finger_val)
        for _ in range(3):
            record_step(target_q6, finger_val)

    # Execute 4-stage grasp
    move_smooth(q_pre, GRIPPER_OPEN_FINGER, steps=int(rng.integers(20, 27)))
    move_smooth(q_grasp, GRIPPER_OPEN_FINGER, steps=int(rng.integers(16, 22)))
    for _ in range(16):
        record_step(q_grasp, GRIPPER_CLOSE_FINGER)
    move_smooth(q_lift, GRIPPER_CLOSE_FINGER, steps=int(rng.integers(18, 25)))
    for _ in range(8):
        record_step(q_lift, GRIPPER_CLOSE_FINGER)

    if not bool(info["success"][0].item()):
        return None

    return {
        "state": np.stack(states, axis=0),        # (T, 7) float32
        "action": np.stack(actions, axis=0),      # (T, 7) float32
        "global_rgb": np.stack(g_imgs, axis=0),   # (T, H, W, 3) uint8
        "wrist_rgb": np.stack(w_imgs, axis=0),    # (T, H, W, 3) uint8
        "task": TASK_TEXT,
    }


def main():
    parser = argparse.ArgumentParser(description="Collect domain-randomized PiperXPickFoam demos")
    parser.add_argument("--num-episodes", type=int, default=100, help="Target number of successful episodes")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(SIM2REAL_ROOT / "data" / "raw_episodes"),
        help="Output directory for .npz episodes",
    )
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(
        "PiperXPickFoam-v1",
        obs_mode="rgbd",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        sim_backend="cpu",
        domain_randomize=True,
    )
    env.reset(seed=args.seed)
    robot = env.unwrapped.agent.robot
    pm = robot.create_pinocchio_model()
    tcp_idx = [l.name for l in robot.get_links()].index("tcp_link")
    qlimits = robot.get_qlimits()[0].cpu().numpy()

    rng = np.random.default_rng(args.seed)
    saved = 0
    attempts = 0
    t0 = time.monotonic()

    print(f"Collecting {args.num_episodes} successful episodes into {out_dir} ...")
    while saved < args.num_episodes and attempts < args.num_episodes * 3:
        ep = collect_one_episode(env, pm, tcp_idx, qlimits, seed=args.seed + attempts, rng=rng)
        attempts += 1
        if ep is not None:
            ep_path = out_dir / f"ep_{saved:04d}.npz"
            np.savez_compressed(ep_path, **ep)
            saved += 1
            if saved % 10 == 0 or saved == args.num_episodes:
                dt = time.monotonic() - t0
                print(
                    f"  [{saved:3d}/{args.num_episodes}] attempts={attempts} "
                    f"rate={saved/attempts*100:.1f}% elapsed={dt:.1f}s ({dt/saved:.2f}s/ep)"
                )

    env.close()
    print(f"Done! Saved {saved} episodes ({attempts} attempts) to {out_dir}")


if __name__ == "__main__":
    main()
