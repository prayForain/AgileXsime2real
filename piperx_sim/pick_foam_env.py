"""ManiSkill 3 tabletop environment: PiperXPickFoam-v1.

Task: Grasp a 10cm x 5cm x 5cm matte black foam block across its 5cm narrow side
and lift it at least 8 cm above the tabletop.
Includes mounted wrist camera (aligned with Orbbec Dabai DC1), fixed global camera,
and optional visual domain randomization for Sim-to-Real transfer.
"""

from typing import Any

import numpy as np
import sapien
import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from .piper_x_agent import PiperXAgent


@register_env("PiperXPickFoam-v1", max_episode_steps=200)
class PiperXPickFoamEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["piper_x"]
    agent: PiperXAgent

    def __init__(
        self,
        *args,
        robot_uids="piper_x",
        # Real foam dimensions: length 10cm (X), width 5cm (Y), height 5cm (Z)
        foam_half_size: tuple[float, float, float] = (0.05, 0.025, 0.025),
        spawn_x_range: tuple[float, float] = (0.22, 0.35),
        spawn_y_range: tuple[float, float] = (-0.12, 0.12),
        lift_height: float = 0.08,
        robot_init_qpos_noise: float = 0.03,
        domain_randomize: bool = False,
        **kwargs,
    ):
        self.foam_half_size = np.array(foam_half_size, dtype=np.float32)
        self.spawn_x_range = spawn_x_range
        self.spawn_y_range = spawn_y_range
        self.lift_height = lift_height
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.domain_randomize = domain_randomize
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        # Aligned to the real Orbbec Gemini 336L right-front desk view
        global_pose = sapien_utils.look_at(
            eye=[0.24, -0.68, 0.40],
            target=[0.06, 0.06, 0.19],
        )
        return [
            CameraConfig(
                uid="global_camera",
                pose=global_pose,
                width=320,
                height=240,
                fov=np.deg2rad(58.0),
                near=0.01,
                far=5.0,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(
            eye=[0.24, -0.68, 0.40],
            target=[0.06, 0.06, 0.19],
        )
        return CameraConfig(
            uid="render_camera",
            pose=pose,
            width=640,
            height=480,
            fov=np.deg2rad(58.0),
            near=0.01,
            far=5.0,
        )

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0.0, 0.0, 0.0]))

    def _load_scene(self, options: dict):
        # Tabletop at z = 0.0 (light gray/off-white desk surface)
        table_builder = self.scene.create_actor_builder()
        table_half = [0.60, 0.60, 0.025]
        table_mat = sapien.render.RenderMaterial(
            base_color=[0.84, 0.85, 0.86, 1.0],
            roughness=0.75,
            metallic=0.0,
        )
        table_builder.add_box_visual(half_size=table_half, material=table_mat)
        table_builder.add_box_collision(half_size=table_half)
        table_builder.initial_pose = sapien.Pose(p=[0.35, 0.0, -table_half[2]])
        self.table = table_builder.build_static(name="table")

        # Light office backdrop wall behind the desk (eliminates pitch-black sim void)
        wall_builder = self.scene.create_actor_builder()
        wall_mat = sapien.render.RenderMaterial(
            base_color=[0.76, 0.78, 0.77, 1.0],
            roughness=0.85,
            metallic=0.0,
        )
        wall_builder.add_box_visual(half_size=[1.5, 0.02, 1.0], material=wall_mat)
        wall_builder.initial_pose = sapien.Pose(p=[0.1, 0.75, 0.60])
        self.back_wall = wall_builder.build_static(name="back_wall")

        # Matte black foam block: 10cm x 5cm x 5cm, lightweight & high friction
        foam_builder = self.scene.create_actor_builder()
        foam_phys = sapien.physx.PhysxMaterial(
            static_friction=1.6,
            dynamic_friction=1.6,
            restitution=0.0,
        )
        foam_mat = sapien.render.RenderMaterial(
            base_color=[0.07, 0.07, 0.08, 1.0],
            roughness=0.92,
            metallic=0.0,
        )
        foam_builder.add_box_visual(
            half_size=self.foam_half_size.tolist(),
            material=foam_mat,
        )
        foam_builder.add_box_collision(
            half_size=self.foam_half_size.tolist(),
            material=foam_phys,
            density=160.0,  # ~40g for 10x5x5 cm foam block
        )
        foam_builder.initial_pose = sapien.Pose(
            p=[0.28, 0.0, float(self.foam_half_size[2])]
        )
        self.foam = foam_builder.build(name="black_foam")

    def _apply_visual_randomization(self):
        """Lightweight per-episode visual domain randomization (CPU/single-env)."""
        rng = np.random.default_rng()
        # Randomize tabletop brightness & subtle tint (gray / off-white / warm desk)
        base_gray = float(rng.uniform(0.68, 0.94))
        tint = rng.uniform(-0.04, 0.04, size=3)
        t_col = np.clip(base_gray + tint, 0.55, 0.98).tolist() + [1.0]
        for obj in self.table._objs:
            comp = obj.find_component_by_type(sapien.render.RenderBodyComponent)
            if comp is not None:
                for shape in comp.render_shapes:
                    for part in shape.parts:
                        part.material.base_color = t_col
                        part.material.roughness = float(rng.uniform(0.55, 0.92))

        # Randomize foam shade (deep black to dark charcoal)
        f_val = float(rng.uniform(0.03, 0.14))
        f_col = [f_val, f_val, f_val + 0.01, 1.0]
        for obj in self.foam._objs:
            comp = obj.find_component_by_type(sapien.render.RenderBodyComponent)
            if comp is not None:
                for shape in comp.render_shapes:
                    for part in shape.parts:
                        part.material.base_color = f_col
                        part.material.roughness = float(rng.uniform(0.80, 0.98))

        # Randomize back wall shade (office white/gray/beige tones)
        w_gray = float(rng.uniform(0.55, 0.88))
        w_tint = rng.uniform(-0.06, 0.06, size=3)
        w_col = np.clip(w_gray + w_tint, 0.40, 0.95).tolist() + [1.0]
        for obj in self.back_wall._objs:
            comp = obj.find_component_by_type(sapien.render.RenderBodyComponent)
            if comp is not None:
                for shape in comp.render_shapes:
                    for part in shape.parts:
                        part.material.base_color = w_col

        # Randomize ambient light intensity
        for sub_scene in self.scene.sub_scenes:
            amb = float(rng.uniform(0.25, 0.60))
            sub_scene.ambient_light = [amb, amb, amb]

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            qpos = np.array(
                [0.0, 0.70, -0.85, 0.70, 0.0, 0.0, 0.045, 0.045],
                dtype=np.float32,
            )
            qpos_t = torch.tensor(qpos, device=self.device).unsqueeze(0).repeat(b, 1)
            if self.robot_init_qpos_noise > 0:
                qpos_t[:, :6] += (
                    torch.randn((b, 6), device=self.device) * self.robot_init_qpos_noise
                )
            self.agent.reset(qpos_t)
            self.agent.robot.set_pose(sapien.Pose(p=[0.0, 0.0, 0.0]))

            # Spawn black foam block at random (x, y, yaw) on tabletop
            xyz = torch.zeros((b, 3), device=self.device)
            xyz[:, 0] = (
                torch.rand((b,), device=self.device)
                * (self.spawn_x_range[1] - self.spawn_x_range[0])
                + self.spawn_x_range[0]
            )
            xyz[:, 1] = (
                torch.rand((b,), device=self.device)
                * (self.spawn_y_range[1] - self.spawn_y_range[0])
                + self.spawn_y_range[0]
            )
            xyz[:, 2] = float(self.foam_half_size[2])
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.foam.set_pose(Pose.create_from_pq(xyz, qs))

        if self.domain_randomize:
            self._apply_visual_randomization()

    def evaluate(self) -> dict[str, Any]:
        is_grasped = self.agent.is_grasping(self.foam)
        foam_z = self.foam.pose.p[:, 2]
        is_lifted = foam_z >= (float(self.foam_half_size[2]) + self.lift_height)
        success = torch.logical_and(is_grasped, is_lifted)
        return {
            "is_grasped": is_grasped,
            "is_lifted": is_lifted,
            "foam_z": foam_z,
            "success": success,
        }

    def _get_obs_extra(self, info: dict):
        return {
            "tcp_pose": self.agent.tcp_pose.raw_pose,
            "foam_pose": self.foam.pose.raw_pose,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        tcp_to_foam = torch.linalg.norm(
            self.foam.pose.p - self.agent.tcp_pos, axis=1
        )
        reaching = 1.0 - torch.tanh(5.0 * tcp_to_foam)
        grasping = info["is_grasped"].float()
        lift_progress = torch.clamp(
            (self.foam.pose.p[:, 2] - float(self.foam_half_size[2])) / self.lift_height,
            0.0,
            1.0,
        )
        return reaching + grasping + 2.0 * grasping * lift_progress

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs, action, info) / 4.0
