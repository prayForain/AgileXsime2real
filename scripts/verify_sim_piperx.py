#!/usr/bin/env python3
"""Verify PiperXAgent inside ManiSkill 3:
  1. Loads PiperX + AgxGripper URDF cleanly on GPU/CPU
  2. Cross-checks simulation flange FK vs real pyAgxArm FK at zero and stretched poses
  3. Renders zero pose, ready pose (gripper open & closed), and wrist_camera view
"""

import math
import os
import sys
from pathlib import Path

import cv2
import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import sapien
import torch
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils

# Add sim2real root to sys.path so piperx_sim registers PiperXAgent
SIM2REAL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SIM2REAL_ROOT))
import piperx_sim  # noqa: E402, F401

from pyAgxArm import AgxArmFactory, ArmModel, PiperFW, create_agx_arm_config  # noqa: E402

DEBUG_DIR = SIM2REAL_ROOT / "debug"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)


def save_rgb(tensor_or_arr, path: Path):
    img = tensor_or_arr
    if hasattr(img, "cpu"):
        img = img.cpu().numpy()
    if img.ndim == 4:
        img = img[0]
    if img.dtype != np.uint8:
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"Saved: {path}  shape={img.shape}")


def main():
    # Create an EmptyEnv with PiperXAgent and a nice side/front overview camera
    env = gym.make(
        "Empty-v1",
        robot_uids="piper_x",
        obs_mode="rgbd",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        sensor_configs=dict(shader_pack="default"),
        human_render_camera_configs=dict(
            pose=sapien_utils.look_at(eye=[0.65, -0.55, 0.55], target=[0.12, 0.0, 0.22]),
            width=640,
            height=480,
        ),
    )

    obs, _ = env.reset(seed=0)
    agent = env.unwrapped.agent
    robot = agent.robot
    print("Loaded PiperXAgent successfully!")
    print("Active joints:", [j.name for j in robot.active_joints])
    print("Action space :", env.action_space)

    # Compare ManiSkill flange_link FK vs pyAgxArm SDK FK (no CAN connection needed!)
    sdk_cfg = create_agx_arm_config(
        robot=ArmModel.PIPER_X,
        firmeware_version=PiperFW.V189,
        interface="socketcan",
        channel="can0",
    )
    sdk_robot = AgxArmFactory.create_arm(sdk_cfg)

    flange_link = robot.links_map["flange_link"]
    link6 = robot.links_map["link6"]

    test_poses = [
        ("zero", [0, 0, 0, 0, 0, 0]),
        ("upward_stretch", [0, 50, -55, 0, 0, 0]),
        ("look_left", [18, 50, -55, -15, 20, 25]),
    ]
    print("\n=== FK Cross-Check: ManiSkill URDF vs pyAgxArm SDK ===")
    for name, degs in test_poses:
        rads = [math.radians(d) for d in degs]
        qpos = np.array(rads + [0.0, 0.0], dtype=np.float32)
        robot.set_qpos(qpos)
        robot.set_qvel(np.zeros_like(qpos))
        # Set PD target to match qpos so scene.step() holds the exact pose
        env.step(np.array(rads + [0.0], dtype=np.float32))
        sim_p6 = link6.pose.p[0].cpu().numpy()
        sim_pf = flange_link.pose.p[0].cpu().numpy()
        sdk_p = np.array(sdk_robot.fk(rads)[:3])
        err_mm = np.linalg.norm(sim_pf - sdk_p) * 1000.0
        print(
            f"  {name:<16} | SDK={np.round(sdk_p,4)} | Sim(flange)={np.round(sim_pf,4)} "
            f"| Sim(link6)={np.round(sim_p6,4)} | diff={err_mm:.2f} mm"
        )

    # 1) Render zero pose
    robot.set_qpos(agent.keyframes["zero"].qpos)
    action_zero = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    for _ in range(10):
        obs, _, _, _, _ = env.step(action_zero)
    save_rgb(env.render(), DEBUG_DIR / "sim_zero_pose.jpg")

    # 2) Move via PD controller to 'ready' with gripper OPEN (0.045m/finger = 90mm width)
    ready_arm = [0.0, math.radians(55), math.radians(-65), math.radians(35), 0.0, 0.0]
    action_open = np.array(ready_arm + [0.045], dtype=np.float32)
    robot.set_qpos(np.array(ready_arm + [0.045, 0.045], dtype=np.float32))
    for _ in range(25):
        obs, _, _, _, _ = env.step(action_open)
    save_rgb(env.render(), DEBUG_DIR / "sim_ready_open.jpg")
    save_rgb(obs["sensor_data"]["wrist_camera"]["rgb"], DEBUG_DIR / "sim_wrist_open.jpg")

    # 3) Close gripper via PD controller (0.0m/finger = 0mm closed)
    action_closed = np.array(ready_arm + [0.0], dtype=np.float32)
    for _ in range(30):
        obs, _, _, _, _ = env.step(action_closed)
    save_rgb(env.render(), DEBUG_DIR / "sim_ready_closed.jpg")
    save_rgb(obs["sensor_data"]["wrist_camera"]["rgb"], DEBUG_DIR / "sim_wrist_closed.jpg")

    q_final = robot.get_qpos()[0].cpu().numpy()
    print("\nFinal qpos (arm deg + fingers mm):",
          [round(math.degrees(x), 1) for x in q_final[:6]],
          [round(x * 1000, 1) for x in q_final[6:]])

    env.close()
    print("\nPhase 1 verification complete!")


if __name__ == "__main__":
    main()
