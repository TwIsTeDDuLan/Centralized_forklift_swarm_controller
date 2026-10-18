"""Direct-workflow go-to-pose environment for forklift_b.

Every quantity the actor sees can be rebuilt on the ROS side from the overhead camera alone (tag x, y, theta)
and the last command the node sent, so the trained actor runs unchanged in ``rl_forklift_controller.py``.
The observation layout is mirrored there; keep both in sync (``build_actor_obs`` in that file).

Conventions (2D, z up, radians):
  * yaw   : heading of the body +X axis (counterweight side). The forks point along yaw + pi.
  * goal  : position for the reference point (the ArUco tag) and, optionally, a target body yaw.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import sample_uniform, wrap_to_pi

from ..robot import DRIVE_JOINT, STEER_JOINT

if TYPE_CHECKING:
    from .forklift_nav_env_cfg import ForkliftNavEnvCfg


def rotate_2d(x: torch.Tensor, y: torch.Tensor, angle: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    c, s = torch.cos(angle), torch.sin(angle)
    return c * x - s * y, s * x + c * y


class ForkliftNavEnv(DirectRLEnv):
    cfg: ForkliftNavEnvCfg

    def __init__(self, cfg: ForkliftNavEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.robot = self.scene["robot"]
        drive_ids, _ = self.robot.find_joints(DRIVE_JOINT)
        steer_ids, _ = self.robot.find_joints(STEER_JOINT)
        self._drive_idx = torch.tensor(drive_ids, device=self.device)
        self._steer_idx = torch.tensor(steer_ids, device=self.device)

        self._speed_scale = math.radians(self.cfg.max_wheel_speed_deg)  # action 1.0 -> rad/s
        self._steer_scale = math.radians(self.cfg.max_steer_deg)  # action 1.0 -> rad
        self._ref_offset = torch.tensor(self.cfg.ref_offset_b, device=self.device)

        n = self.num_envs
        self.actions = torch.zeros(n, 2, device=self.device)
        self.prev_actions = torch.zeros(n, 2, device=self.device)
        self.applied_actions = torch.zeros(n, 2, device=self.device)
        self.delayed = torch.zeros(n, dtype=torch.bool, device=self.device)

        self.goal_pos = torch.zeros(n, 2, device=self.device)  # world frame, per-env origin included
        self.goal_yaw = torch.zeros(n, device=self.device)
        self.has_heading = torch.zeros(n, device=self.device)
        self.start_dist = torch.ones(n, device=self.device)
        self.prev_potential = torch.zeros(n, device=self.device)

        self.reached = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.failed = torch.zeros(n, dtype=torch.bool, device=self.device)

        self.curriculum_level = 0.0
        self.success_ema = 0.0

        self._episode_sums = {
            k: torch.zeros(n, device=self.device) for k in ("progress", "success", "failure", "time", "action", "action_rate")
        }

    # ------ state helpers ------
    def _pose_2d(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Reference point position (world xy) and body yaw."""
        root_pos = self.robot.data.root_pos_w.torch[:, :2]
        yaw = self.robot.data.heading_w.torch
        ox, oy = rotate_2d(self._ref_offset[0].expand_as(yaw), self._ref_offset[1].expand_as(yaw), yaw)
        return root_pos + torch.stack((ox, oy), dim=-1), yaw

    def _goal_errors(self, ref_pos: torch.Tensor, yaw: torch.Tensor):
        delta = self.goal_pos - ref_pos
        dist = torch.linalg.norm(delta, dim=-1)
        gx, gy = rotate_2d(delta[:, 0], delta[:, 1], -yaw)
        heading_err = wrap_to_pi(self.goal_yaw - yaw)
        return gx, gy, dist, heading_err

    def _potential(self, dist: torch.Tensor, heading_err: torch.Tensor) -> torch.Tensor:
        w = self.cfg.heading_potential_weight * self.has_heading
        return -(dist + w * heading_err.abs() * torch.exp(-dist / self.cfg.heading_potential_sigma))

    # ------ actions ------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.prev_actions = self.actions.clone()
        self.actions = actions.clamp(-1.0, 1.0)
        # envs flagged as delayed apply the previous policy output (one 0.1 s step of latency)
        self.applied_actions = torch.where(self.delayed.unsqueeze(-1), self.prev_actions, self.actions)

    def _apply_action(self) -> None:
        speed = self.applied_actions[:, 0:1] * self._speed_scale
        steer = self.applied_actions[:, 1:2] * self._steer_scale
        self.robot.set_joint_velocity_target_index(target=speed, joint_ids=self._drive_idx)
        self.robot.set_joint_position_target_index(target=steer, joint_ids=self._steer_idx)

    # ------ MDP ------
    def _get_observations(self) -> dict:
        ref_pos, yaw = self._pose_2d()
        # camera noise on what the actor sees; the critic gets the clean state
        noisy_pos = ref_pos + self.cfg.pos_noise_std * torch.randn_like(ref_pos)
        noisy_yaw = yaw + self.cfg.yaw_noise_std * torch.randn_like(yaw)
        gx, gy, dist, heading_err = self._goal_errors(noisy_pos, noisy_yaw)

        s = self.cfg.pos_obs_scale
        clip = self.cfg.pos_obs_clip
        # keep the direction when clipping far goals
        far = torch.clamp(dist * s / clip, min=1.0)
        actor = torch.stack(
            (
                gx * s / far,
                gy * s / far,
                torch.clamp(dist * s, max=clip),
                torch.sin(heading_err) * self.has_heading,
                torch.cos(heading_err) * self.has_heading,
                self.has_heading,
                self.actions[:, 0],
                self.actions[:, 1],
            ),
            dim=-1,
        )

        lin_vel_b = self.robot.data.root_lin_vel_b.torch[:, :2]
        yaw_rate = self.robot.data.root_ang_vel_w.torch[:, 2:3]
        joint_pos = self.robot.data.joint_pos.torch
        joint_vel = self.robot.data.joint_vel.torch
        _, _, dist_true, heading_err_true = self._goal_errors(ref_pos, yaw)
        critic = torch.cat(
            (
                actor,
                lin_vel_b,
                yaw_rate,
                joint_pos[:, self._steer_idx] / self._steer_scale,
                joint_vel[:, self._drive_idx] / self._speed_scale,
                torch.stack((dist_true * s, heading_err_true * self.has_heading), dim=-1),
            ),
            dim=-1,
        )
        return {"policy": actor, "critic": critic}

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        ref_pos, yaw = self._pose_2d()
        _, _, dist, heading_err = self._goal_errors(ref_pos, yaw)
        speed = torch.linalg.norm(self.robot.data.root_lin_vel_b.torch[:, :2], dim=-1)

        heading_ok = (heading_err.abs() < self.cfg.heading_tolerance) | (self.has_heading < 0.5)
        self.reached = (dist < self.cfg.pos_tolerance) & heading_ok & (speed < self.cfg.stop_speed)
        tipped = self.robot.data.projected_gravity_b.torch[:, 2] > -0.7
        self.failed = (dist > self.start_dist + self.cfg.max_overshoot) | tipped

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return self.reached | self.failed, time_out

    def _get_rewards(self) -> torch.Tensor:
        ref_pos, yaw = self._pose_2d()
        _, _, dist, heading_err = self._goal_errors(ref_pos, yaw)
        potential = self._potential(dist, heading_err)

        terms = {
            "progress": self.cfg.rew_progress * (potential - self.prev_potential),
            "success": self.cfg.rew_success * self.reached.float(),
            "failure": self.cfg.rew_failure * self.failed.float(),
            "time": torch.full_like(dist, self.cfg.rew_time),
            "action": self.cfg.rew_action * torch.sum(self.actions**2, dim=-1),
            "action_rate": self.cfg.rew_action_rate * torch.sum((self.actions - self.prev_actions) ** 2, dim=-1),
        }
        self.prev_potential = potential
        reward = torch.zeros_like(dist)
        for k, v in terms.items():
            self._episode_sums[k] += v
            reward += v
        return reward

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device)

        # ------ logging + curriculum (skip the very first reset of all envs) ------
        if self.common_step_counter > 0:
            success = self.reached[env_ids].float().mean()
            log = self.extras.setdefault("log", {})
            for k, v in self._episode_sums.items():
                log[f"Episode_Reward/{k}"] = v[env_ids].mean()
            log["Metrics/success_rate"] = success
            log["Metrics/failure_rate"] = self.failed[env_ids].float().mean()
            log["Metrics/curriculum_level"] = self.curriculum_level
            log["Metrics/goal_dist_max"] = self._goal_dist_max()

            self.success_ema = 0.95 * self.success_ema + 0.05 * success.item()
            if self.success_ema > self.cfg.curriculum_success_threshold:
                self.curriculum_level = min(1.0, self.curriculum_level + self.cfg.curriculum_step)
        for v in self._episode_sums.values():
            v[env_ids] = 0.0

        super()._reset_idx(env_ids)
        n = len(env_ids)

        # ------ robot: random yaw at the env origin, joints at rest ------
        yaw = sample_uniform(-math.pi, math.pi, (n,), self.device)
        root_pose = self.robot.data.default_root_pose.torch[env_ids].clone()
        root_pose[:, :3] += self.scene.env_origins[env_ids]
        root_pose[:, 3:7] = 0.0
        root_pose[:, 5] = torch.sin(yaw / 2)  # (x, y, z, w) quaternion
        root_pose[:, 6] = torch.cos(yaw / 2)
        root_vel = torch.zeros(n, 6, device=self.device)
        joint_pos = self.robot.data.default_joint_pos.torch[env_ids].clone()
        joint_pos[:, self._steer_idx] = sample_uniform(-0.5, 0.5, (n, 1), self.device) * self._steer_scale
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=root_vel, env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)

        # ------ goal relative to the reference point ------
        ox, oy = rotate_2d(self._ref_offset[0].expand(n), self._ref_offset[1].expand(n), yaw)
        ref_pos = root_pose[:, :2] + torch.stack((ox, oy), dim=-1)
        r = sample_uniform(self.cfg.goal_dist_min, self._goal_dist_max(), (n,), self.device)
        phi = sample_uniform(-math.pi, math.pi, (n,), self.device)
        self.goal_pos[env_ids] = ref_pos + torch.stack((r * torch.cos(phi), r * torch.sin(phi)), dim=-1)
        self.goal_yaw[env_ids] = sample_uniform(-math.pi, math.pi, (n,), self.device)
        self.has_heading[env_ids] = (torch.rand(n, device=self.device) < self.cfg.heading_goal_prob).float()
        self.start_dist[env_ids] = r

        self.delayed[env_ids] = torch.rand(n, device=self.device) < self.cfg.action_delay_prob
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.applied_actions[env_ids] = 0.0

        heading_err = wrap_to_pi(self.goal_yaw[env_ids] - yaw)
        w = self.cfg.heading_potential_weight * self.has_heading[env_ids]
        self.prev_potential[env_ids] = -(r + w * heading_err.abs() * torch.exp(-r / self.cfg.heading_potential_sigma))

    def _goal_dist_max(self) -> float:
        c = self.cfg
        return c.goal_dist_max_start + self.curriculum_level * (c.goal_dist_max_end - c.goal_dist_max_start)

    # ------ debug visualization ------
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_markers"):
                # simple shapes only: no Nucleus downloads
                cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/forklift_goal",
                    markers={
                        "heading": sim_utils.CuboidCfg(
                            size=(1.2, 0.08, 0.08),
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.9, 0.1)),
                        ),
                        "any_heading": sim_utils.SphereCfg(
                            radius=0.2,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.6, 0.1)),
                        ),
                    },
                )
                self.goal_markers = VisualizationMarkers(cfg)
            self.goal_markers.set_visibility(True)
        elif hasattr(self, "goal_markers"):
            self.goal_markers.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not hasattr(self, "goal_markers") or not hasattr(self, "goal_pos"):
            return
        # the bar points the way the FORKS should face (body yaw + pi), centred ahead of the goal point
        fork_yaw = self.goal_yaw + math.pi
        pos = torch.zeros(self.num_envs, 3, device=self.device)
        pos[:, 0] = self.goal_pos[:, 0] + 0.6 * torch.cos(fork_yaw) * self.has_heading
        pos[:, 1] = self.goal_pos[:, 1] + 0.6 * torch.sin(fork_yaw) * self.has_heading
        pos[:, 2] = 0.1
        quat = torch.zeros(self.num_envs, 4, device=self.device)
        quat[:, 2] = torch.sin(fork_yaw / 2)
        quat[:, 3] = torch.cos(fork_yaw / 2)
        self.goal_markers.visualize(pos, quat, marker_indices=(1.0 - self.has_heading).long())
