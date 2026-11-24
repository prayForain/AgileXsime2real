"""ManiSkill 3 BaseAgent definition for AgileX PiperX + AgxGripper + Wrist Camera."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import sapien
import torch

from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import (
    PDJointPosControllerConfig,
    PDJointPosMimicControllerConfig,
    deepcopy_dict,
)
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.actor import Actor

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
PIPER_X_URDF = str(ASSETS_DIR / "piper_x_with_gripper.urdf")


@register_agent()
class PiperXAgent(BaseAgent):
    uid = "piper_x"
    urdf_path = PIPER_X_URDF
    fix_root_link = True

    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link=dict(
            gripper_link1=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            gripper_link2=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
        ),
    )

    keyframes = dict(
        # Official PiperX folded zero posture [0, 0, 0, 0, 0, 0] + closed gripper
        zero=Keyframe(
            qpos=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            pose=sapien.Pose(),
        ),
        # Upright tabletop ready-to-grasp posture (open gripper: 0.04m per finger = 8cm width)
        ready=Keyframe(
            qpos=np.array(
                [0.0, 0.85, -0.95, 0.65, 0.0, 0.0, 0.04, 0.04],
                dtype=np.float32,
            ),
            pose=sapien.Pose(),
        ),
    )

    arm_joint_names = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
    ]
    gripper_joint_names = [
        "gripper_joint1",
        "gripper_joint2",
    ]
    ee_link_name = "tcp_link"

    arm_stiffness = 1e3
    arm_damping = 1e2
    arm_force_limit = 100.0

    gripper_stiffness = 1e3
    gripper_damping = 1e2
    gripper_force_limit = 50.0

    @property
    def _controller_configs(self):
        # Absolute joint position control (direct 1:1 mapping to real PiperX move_j in radians)
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            use_delta=True,
            use_target=False,
        )
        arm_pd_joint_target_delta_pos = deepcopy(arm_pd_joint_delta_pos)
        arm_pd_joint_target_delta_pos.use_target = True

        # Symmetric parallel gripper: single 1D action in [0.0, 0.05] per finger (total width 0..0.10m)
        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            lower=0.0,
            upper=0.05,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            normalize_action=False,
            mimic={"gripper_joint2": {"joint": "gripper_joint1"}},
        )
        gripper_pd_joint_pos_norm = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            lower=0.0,
            upper=0.05,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            normalize_action=True,
            mimic={"gripper_joint2": {"joint": "gripper_joint1"}},
        )

        controller_configs = dict(
            pd_joint_pos=dict(
                arm=arm_pd_joint_pos,
                gripper=gripper_pd_joint_pos,
            ),
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos,
                gripper=gripper_pd_joint_pos_norm,
            ),
            pd_joint_target_delta_pos=dict(
                arm=arm_pd_joint_target_delta_pos,
                gripper=gripper_pd_joint_pos_norm,
            ),
        )
        return deepcopy_dict(controller_configs)

    @property
    def _sensor_configs(self):
        # Mounted above gripper_base looking forward & slightly down toward the fingertips,
        # matching the Orbbec Dabai DC1 wrist camera framing.
        # In gripper_base frame: +Z is forward along fingers, +/-Y is finger opening axis.
        wrist_pose = sapien_utils.look_at(
            eye=[-0.065, 0.0, 0.035],
            target=[0.0, 0.0, 0.22],
            up=[-1.0, 0.0, 0.0],
        )
        return [
            CameraConfig(
                uid="wrist_camera",
                pose=wrist_pose,
                width=320,
                height=240,
                fov=np.deg2rad(58.0),
                near=0.01,
                far=5.0,
                mount=self.robot.links_map["gripper_base"],
            )
        ]

    def _after_loading_articulation(self):
        self.finger1_link = self.robot.links_map["gripper_link1"]
        self.finger2_link = self.robot.links_map["gripper_link2"]
        self.tcp = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.ee_link_name
        )

    @property
    def tcp_pos(self):
        return self.tcp.pose.p

    @property
    def tcp_pose(self):
        return self.tcp.pose

    def is_grasping(self, obj: Actor, min_force: float = 0.2, max_angle: float = 85.0):
        l_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger1_link, obj
        )
        r_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger2_link, obj
        )
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)
        # Force exerted BY obj ON finger points outward along +Z (opening direction)
        ldirection = self.finger1_link.pose.to_transformation_matrix()[..., :3, 2]
        rdirection = self.finger2_link.pose.to_transformation_matrix()[..., :3, 2]
        langle = common.compute_angle_between(ldirection, l_contact_forces)
        rangle = common.compute_angle_between(rdirection, r_contact_forces)
        lflag = torch.logical_and(
            lforce >= min_force, torch.rad2deg(langle) <= max_angle
        )
        rflag = torch.logical_and(
            rforce >= min_force, torch.rad2deg(rangle) <= max_angle
        )
        return torch.logical_and(lflag, rflag)

    def is_static(self, threshold: float = 0.2):
        qvel = self.robot.get_qvel()[..., :-2]
        return torch.max(torch.abs(qvel), 1)[0] <= threshold
