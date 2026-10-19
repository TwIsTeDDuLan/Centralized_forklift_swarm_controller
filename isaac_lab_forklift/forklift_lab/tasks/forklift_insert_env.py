"""Direct-workflow fork-insertion environment for forklift_b.

The actor sees only a fork-mounted 3x64 range scan, its last command, and the lift joint — the
same vector the ROS node builds from /forklift_N/scan and /forklift_N/mast_pos. Keep
``build_insert_obs`` in ``rl_policy.py`` and ``OBS_LAYOUT`` in ``export_policy.py`` in sync.

Pallet frame (fixed in each env): origin at the entry-face centre on the floor, +X into the
pockets, +Y left along the 1.20 m face, +Z up. See ``forklift_lab.pallet``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import quat_apply, sample_uniform, wrap_to_pi

from ..pallet import (
    INSERT_TARGET_X,
    INSERT_TARGET_Y,
    MAX_STACK,
    PALLET_DEPTH,
    PALLET_HEIGHT,
    PALLET_WIDTH,
    POCKET_WIDTH,
    POCKET_Y,
    POCKET_Z_CENTER,
    POCKET_Z_HI,
    POCKET_Z_LO,
)
from ..robot import DRIVE_JOINT, LIFT_BODY, LIFT_JOINT, LIFT_LIMITS, STEER_JOINT

if TYPE_CHECKING:
    from .forklift_insert_env_cfg import ForkliftInsertEnvCfg


def _yaw_quat(yaw: torch.Tensor) -> torch.Tensor:
    """(N,) yaw -> (N, 4) quaternion (x, y, z, w)."""
    q = torch.zeros(yaw.shape[0], 4, device=yaw.device, dtype=yaw.dtype)
    q[:, 2] = torch.sin(yaw * 0.5)
    q[:, 3] = torch.cos(yaw * 0.5)
    return q


def raycast_aabbs(
    origins: torch.Tensor,
    directions: torch.Tensor,
    bmin: torch.Tensor,
    bmax: torch.Tensor,
    max_dist: float,
) -> torch.Tensor:
    """Batched ray vs AABB. origins/dirs (N,R,3), boxes (N,B,3) -> ranges (N,R)."""
    o = origins.unsqueeze(2)
    d = directions.unsqueeze(2)
    inv = torch.where(d.abs() > 1e-8, 1.0 / d, torch.full_like(d, 1e8))
    t1 = (bmin.unsqueeze(1) - o) * inv
    t2 = (bmax.unsqueeze(1) - o) * inv
    tmin = torch.minimum(t1, t2).amax(dim=-1)
    tmax = torch.maximum(t1, t2).amin(dim=-1)
    hit = (tmax >= tmin) & (tmax > 0.0) & (tmin < max_dist)
    t = torch.where(tmin > 0.0, tmin, tmax)
    t = torch.where(hit, t, torch.full_like(t, max_dist))
    return t.amin(dim=-1).clamp(0.0, max_dist)


class ForkliftInsertEnv(DirectRLEnv):
    cfg: ForkliftInsertEnvCfg

    def __init__(self, cfg: ForkliftInsertEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.robot = self.scene["robot"]
        self.face_left = self.scene["face_left"]
        self.face_center = self.scene["face_center"]
        self.face_right = self.scene["face_right"]
        self.stack_1 = self.scene["stack_1"]
        self.stack_2 = self.scene["stack_2"]
        self.contact = self.scene["contact_lift"]

        drive_ids, _ = self.robot.find_joints(DRIVE_JOINT)
        steer_ids, _ = self.robot.find_joints(STEER_JOINT)
        lift_ids, _ = self.robot.find_joints(LIFT_JOINT)
        lift_body_ids, _ = self.robot.find_bodies(LIFT_BODY)
        self._drive_idx = torch.tensor(drive_ids, device=self.device)
        self._steer_idx = torch.tensor(steer_ids, device=self.device)
        self._lift_idx = torch.tensor(lift_ids, device=self.device)
        self._lift_body = int(lift_body_ids[0])

        self._speed_scale = math.radians(self.cfg.max_wheel_speed_deg)
        self._steer_scale = math.radians(self.cfg.max_steer_deg)
        self._lift_speed = self.cfg.max_lift_speed
        self._tip_left = torch.tensor(self.cfg.fork_tip_left_l, device=self.device)
        self._tip_right = torch.tensor(self.cfg.fork_tip_right_l, device=self.device)
        self._lidar_off = torch.tensor(self.cfg.lidar_offset_l, device=self.device)
        self._lidar_fwd = torch.tensor((0.0, -1.0, 0.0), device=self.device)
        self._lidar_right = torch.tensor((-1.0, 0.0, 0.0), device=self.device)
        self._lidar_up = torch.tensor((0.0, 0.0, 1.0), device=self.device)

        ch, az = self.cfg.lidar_channels, self.cfg.lidar_azimuth
        h = math.radians(self.cfg.lidar_h_fov_deg)
        v = math.radians(self.cfg.lidar_v_fov_deg)
        az_angles = torch.linspace(-0.5 * h, 0.5 * h, az, device=self.device)
        el_angles = torch.linspace(-0.5 * v, 0.5 * v, ch, device=self.device)
        el, azm = torch.meshgrid(el_angles, az_angles, indexing="ij")
        self._ray_az = azm.reshape(-1)
        self._ray_el = el.reshape(-1)
        self._n_rays = ch * az

        n = self.num_envs
        self.actions = torch.zeros(n, 3, device=self.device)
        self.prev_actions = torch.zeros(n, 3, device=self.device)
        self.applied_actions = torch.zeros(n, 3, device=self.device)
        self.delayed = torch.zeros(n, dtype=torch.bool, device=self.device)

        self.stack_n = torch.ones(n, dtype=torch.long, device=self.device)
        self.target_level = torch.zeros(n, dtype=torch.long, device=self.device)
        self.prev_potential = torch.zeros(n, device=self.device)
        self.reached = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.failed = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.face_hit = torch.zeros(n, dtype=torch.bool, device=self.device)

        self.curriculum_level = 0.0
        self.success_ema = 0.0
        self._episode_sums = {
            k: torch.zeros(n, device=self.device)
            for k in ("progress", "success", "failure", "face", "time", "action", "action_rate")
        }

    # ------ geometry ------
    def _lift_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        pos = self.robot.data.body_pos_w.torch[:, self._lift_body]
        quat = self.robot.data.body_quat_w.torch[:, self._lift_body]
        return pos, quat

    def _fork_tips(self) -> tuple[torch.Tensor, torch.Tensor]:
        pos, quat = self._lift_pose()
        left = pos + quat_apply(quat, self._tip_left.expand_as(pos))
        right = pos + quat_apply(quat, self._tip_right.expand_as(pos))
        return left, right

    def _pocket_targets(self) -> tuple[torch.Tensor, torch.Tensor]:
        z = POCKET_Z_CENTER + self.target_level.float() * PALLET_HEIGHT
        left = torch.stack(
            (
                torch.full((self.num_envs,), INSERT_TARGET_X, device=self.device),
                torch.full((self.num_envs,), INSERT_TARGET_Y[0], device=self.device),
                z,
            ),
            dim=-1,
        ) + self.scene.env_origins
        right = left.clone()
        right[:, 1] = INSERT_TARGET_Y[1] + self.scene.env_origins[:, 1]
        return left, right

    def _tips_in_pallet(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tips in the env pallet frame (origin = env origin, face at x=0)."""
        left, right = self._fork_tips()
        origin = self.scene.env_origins
        z0 = self.target_level.float() * PALLET_HEIGHT
        lp = left - origin
        rp = right - origin
        lp = lp.clone()
        rp = rp.clone()
        lp[:, 2] = left[:, 2] - z0
        rp[:, 2] = right[:, 2] - z0
        return lp, rp, left, right

    def _in_pocket_opening(self, local: torch.Tensor) -> torch.Tensor:
        y_ok = (local[:, 1] - POCKET_Y[0]).abs() < 0.5 * POCKET_WIDTH
        y_ok = y_ok | ((local[:, 1] - POCKET_Y[1]).abs() < 0.5 * POCKET_WIDTH)
        z_ok = (local[:, 2] > POCKET_Z_LO - 0.02) & (local[:, 2] < POCKET_Z_HI + 0.02)
        x_ok = local[:, 0] > -0.08
        return y_ok & z_ok & x_ok

    def _fully_inserted(self, local: torch.Tensor) -> torch.Tensor:
        y_ok = (local[:, 1] - INSERT_TARGET_Y[0]).abs() < self.cfg.pocket_y_tol
        y_ok = y_ok | ((local[:, 1] - INSERT_TARGET_Y[1]).abs() < self.cfg.pocket_y_tol)
        z_ok = (local[:, 2] - POCKET_Z_CENTER).abs() < self.cfg.pocket_z_tol
        return (local[:, 0] > self.cfg.insert_success_x) & y_ok & z_ok

    def _contact_force(self) -> torch.Tensor:
        data = self.contact.data
        # Isaac Sim PhysX 6.1 has no total-force buffer; use the normal component.
        force = data.net_normal_forces_w
        if force is None:
            return torch.zeros(self.num_envs, device=self.device)
        f = force.torch
        if f.ndim == 3:
            f = f.norm(dim=-1).amax(dim=-1)
        else:
            f = f.reshape(self.num_envs, -1).norm(dim=-1)
        return f

    def _lidar_ranges(self) -> torch.Tensor:
        pos, quat = self._lift_pose()
        origin = pos + quat_apply(quat, self._lidar_off.expand_as(pos))
        fwd = quat_apply(quat, self._lidar_fwd.expand_as(pos))
        right = quat_apply(quat, self._lidar_right.expand_as(pos))
        up = quat_apply(quat, self._lidar_up.expand_as(pos))
        az = self._ray_az.view(1, -1, 1)
        el = self._ray_el.view(1, -1, 1)
        dirs = (
            torch.cos(el) * torch.cos(az) * fwd.unsqueeze(1)
            + torch.cos(el) * torch.sin(az) * right.unsqueeze(1)
            + torch.sin(el) * up.unsqueeze(1)
        )
        dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        bmin, bmax = self._lidar_boxes()
        ranges = raycast_aabbs(origin.unsqueeze(1).expand(-1, self._n_rays, -1), dirs, bmin, bmax, self.cfg.lidar_max_range)
        if self.cfg.lidar_range_noise > 0.0:
            ranges = (ranges + self.cfg.lidar_range_noise * torch.randn_like(ranges)).clamp(0.0, self.cfg.lidar_max_range)
        return ranges

    def _lidar_boxes(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Fixed-count AABBs in world: 3 face boards + floor + top + back + 2 extra decks."""
        n = self.num_envs
        origin = self.scene.env_origins
        z0 = self.target_level.float() * PALLET_HEIGHT
        boxes_min = []
        boxes_max = []

        def add(xmin, xmax, ymin, ymax, zmin, zmax):
            boxes_min.append(torch.stack((xmin, ymin, zmin), dim=-1) + origin)
            boxes_max.append(torch.stack((xmax, ymax, zmax), dim=-1) + origin)

        ones = torch.ones(n, device=self.device)
        # face stringers at the target deck
        for y in (-0.55, 0.0, 0.55):
            add(0.0 * ones, 0.05 * ones, (y - 0.05) * ones, (y + 0.05) * ones, z0, z0 + PALLET_HEIGHT)
        # pocket floor / deck top / back wall
        add(0.0 * ones, PALLET_DEPTH * ones, -0.5 * PALLET_WIDTH * ones, 0.5 * PALLET_WIDTH * ones, z0, z0 + POCKET_Z_LO)
        add(
            0.0 * ones,
            PALLET_DEPTH * ones,
            -0.5 * PALLET_WIDTH * ones,
            0.5 * PALLET_WIDTH * ones,
            z0 + POCKET_Z_HI,
            z0 + PALLET_HEIGHT,
        )
        add(
            (PALLET_DEPTH - 0.04) * ones,
            PALLET_DEPTH * ones,
            -0.5 * PALLET_WIDTH * ones,
            0.5 * PALLET_WIDTH * ones,
            z0,
            z0 + PALLET_HEIGHT,
        )
        # extra solid decks (underground when that level is unused or is the target)
        for level, _ in ((1, self.stack_1), (2, self.stack_2)):
            z = level * PALLET_HEIGHT * ones
            used = (self.stack_n > level) & (self.target_level != level)
            zmin = torch.where(used, z, torch.full_like(z, -3.0))
            zmax = torch.where(used, z + PALLET_HEIGHT, torch.full_like(z, -2.8))
            add(0.0 * ones, PALLET_DEPTH * ones, -0.5 * PALLET_WIDTH * ones, 0.5 * PALLET_WIDTH * ones, zmin, zmax)
        return torch.stack(boxes_min, dim=1), torch.stack(boxes_max, dim=1)

    # ------ actions ------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.prev_actions = self.actions.clone()
        self.actions = actions.clamp(-1.0, 1.0)
        self.applied_actions = torch.where(self.delayed.unsqueeze(-1), self.prev_actions, self.actions)

    def _apply_action(self) -> None:
        speed = self.applied_actions[:, 0:1] * self._speed_scale
        steer = self.applied_actions[:, 1:2] * self._steer_scale
        lift = self.robot.data.joint_pos.torch[:, self._lift_idx]
        lift = (lift + self.applied_actions[:, 2:3] * self._lift_speed * self.cfg.sim.dt).clamp(*LIFT_LIMITS)
        self.robot.set_joint_velocity_target_index(target=speed, joint_ids=self._drive_idx)
        self.robot.set_joint_position_target_index(target=steer, joint_ids=self._steer_idx)
        self.robot.set_joint_position_target_index(target=lift, joint_ids=self._lift_idx)

    # ------ MDP ------
    def _get_observations(self) -> dict:
        ranges = self._lidar_ranges()
        lift = self.robot.data.joint_pos.torch[:, self._lift_idx[0]]
        lo, hi = LIFT_LIMITS
        lift_norm = (lift - lo) / (hi - lo) * 2.0 - 1.0
        actor = torch.cat((ranges / self.cfg.lidar_max_range, self.actions, lift_norm.unsqueeze(-1)), dim=-1)

        lp, rp, _, _ = self._tips_in_pallet()
        tgt_l, tgt_r = self._pocket_targets()
        tips_l, tips_r = self._fork_tips()
        err_l = tips_l - tgt_l
        err_r = tips_r - tgt_r
        depth = 0.5 * (lp[:, 0] + rp[:, 0])
        lat = 0.5 * (lp[:, 1] + rp[:, 1])
        height = 0.5 * (lp[:, 2] + rp[:, 2]) - POCKET_Z_CENTER
        yaw = self.robot.data.heading_w.torch
        # forks point along yaw+pi; desired fork heading is +X (0) so desired body yaw is pi
        yaw_err = wrap_to_pi(yaw - math.pi)
        contact = self._contact_force() / 50.0
        critic = torch.cat(
            (
                actor,
                err_l,
                err_r,
                torch.stack((depth, lat, yaw_err, height), dim=-1),
                contact.unsqueeze(-1),
            ),
            dim=-1,
        )
        return {"policy": actor, "critic": critic}

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        lp, rp, _, _ = self._tips_in_pallet()
        in_open = self._in_pocket_opening(lp) & self._in_pocket_opening(rp)
        force = self._contact_force()
        self.face_hit = (force > self.cfg.face_force_threshold) & (~in_open)
        self.reached = self._fully_inserted(lp) & self._fully_inserted(rp) & (~self.face_hit)
        tipped = self.robot.data.projected_gravity_b.torch[:, 2] > -0.7
        past = (lp[:, 0] > PALLET_DEPTH + 0.35) | (rp[:, 0] > PALLET_DEPTH + 0.35)
        self.failed = self.face_hit | tipped | past
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return self.reached | self.failed, time_out

    def _potential(self) -> torch.Tensor:
        """Align-then-insert potential. Depth only pays once the tines are in front of the pockets.

        The previous Euclidean tip-to-target potential paid for driving in cocked (one tine in,
        one beside the centre stringer). Eval timeouts sat at ~0.45 m depth with that local
        optimum. Gate depth on lateral / yaw / tip-skew so backing up to straighten is better
        than finishing a crooked approach.
        """
        lp, rp, _, _ = self._tips_in_pallet()
        yz_l = (lp[:, 1] - INSERT_TARGET_Y[0]).abs() + (lp[:, 2] - POCKET_Z_CENTER).abs()
        yz_r = (rp[:, 1] - INSERT_TARGET_Y[1]).abs() + (rp[:, 2] - POCKET_Z_CENTER).abs()
        yaw_err = wrap_to_pi(self.robot.data.heading_w.torch - math.pi).abs()
        skew = (lp[:, 0] - rp[:, 0]).abs()
        align = yz_l + yz_r + yaw_err + skew
        gate = torch.exp(-align / 0.18)
        depth = 0.5 * ((lp[:, 0] - INSERT_TARGET_X).abs() + (rp[:, 0] - INSERT_TARGET_X).abs())
        # Always pull tips toward the face while they are still outside, so staging is not idle.
        outside = 0.5 * (lp[:, 0].clamp(max=0.0).abs() + rp[:, 0].clamp(max=0.0).abs())
        return -(2.5 * align + gate * depth + 0.30 * outside)

    def _get_rewards(self) -> torch.Tensor:
        potential = self._potential()
        terms = {
            "progress": self.cfg.rew_progress * (potential - self.prev_potential),
            "success": self.cfg.rew_success * self.reached.float(),
            "failure": self.cfg.rew_failure * (self.failed & ~self.face_hit).float(),
            "face": self.cfg.rew_face_hit * self.face_hit.float(),
            "time": torch.full_like(potential, self.cfg.rew_time),
            "action": self.cfg.rew_action * torch.sum(self.actions**2, dim=-1),
            "action_rate": self.cfg.rew_action_rate * torch.sum((self.actions - self.prev_actions) ** 2, dim=-1),
        }
        self.prev_potential = potential
        reward = torch.zeros_like(potential)
        for k, v in terms.items():
            self._episode_sums[k] += v
            reward += v
        return reward

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device)

        if self.common_step_counter > 0:
            success = self.reached[env_ids].float().mean()
            log = self.extras.setdefault("log", {})
            for k, v in self._episode_sums.items():
                log[f"Episode_Reward/{k}"] = v[env_ids].mean()
            log["Metrics/success_rate"] = success
            log["Metrics/failure_rate"] = self.failed[env_ids].float().mean()
            log["Metrics/face_hit_rate"] = self.face_hit[env_ids].float().mean()
            log["Metrics/curriculum_level"] = self.curriculum_level
            lp, rp, _, _ = self._tips_in_pallet()
            log["Metrics/insert_depth"] = 0.5 * (lp[env_ids, 0] + rp[env_ids, 0]).mean()
            log["Metrics/min_tip_x"] = torch.minimum(lp[env_ids, 0], rp[env_ids, 0]).mean()
            log["Metrics/abs_tip_y"] = 0.5 * (lp[env_ids, 1].abs() + rp[env_ids, 1].abs()).mean()
            log["Metrics/lat_err"] = (0.5 * (lp[env_ids, 1] + rp[env_ids, 1])).abs().mean()
            log["Metrics/yaw_err"] = wrap_to_pi(self.robot.data.heading_w.torch[env_ids] - math.pi).abs().mean()
            log["Metrics/tip_skew"] = (lp[env_ids, 0] - rp[env_ids, 0]).abs().mean()
            log["Metrics/tip_z_err"] = (0.5 * (lp[env_ids, 2] + rp[env_ids, 2]) - POCKET_Z_CENTER).abs().mean()
            self.success_ema = 0.95 * self.success_ema + 0.05 * success.item()
            # Advance slowly (once per ~50 env steps) so easy inserts stay in the mix longer.
            if self.success_ema > self.cfg.curriculum_success_threshold and self.common_step_counter % 50 == 0:
                self.curriculum_level = min(1.0, self.curriculum_level + self.cfg.curriculum_step)
        for v in self._episode_sums.values():
            v[env_ids] = 0.0

        super()._reset_idx(env_ids)
        n = len(env_ids)
        c = self.curriculum_level
        lat_span = self.cfg.staging_lat_start + c * (self.cfg.staging_lat_end - self.cfg.staging_lat_start)
        yaw_span = self.cfg.staging_yaw_start + c * (self.cfg.staging_yaw_end - self.cfg.staging_yaw_start)
        lift_span = self.cfg.lift_err_start + c * (self.cfg.lift_err_end - self.cfg.lift_err_start)
        stack_p = self.cfg.stack_prob_start + c * (self.cfg.stack_prob_end - self.cfg.stack_prob_start)

        # Keep a slice of easy staging even at curriculum 1 so the policy does not forget
        # a clean two-tine insert while it practices recoveries.
        easy = torch.rand(n, device=self.device) < 0.32
        lat_hi = torch.where(easy, self.cfg.staging_lat_start, lat_span)
        yaw_hi = torch.where(easy, self.cfg.staging_yaw_start, yaw_span)
        lift_hi = torch.where(easy, self.cfg.lift_err_start, lift_span)

        stack = torch.ones(n, dtype=torch.long, device=self.device)
        extra = (~easy) & (torch.rand(n, device=self.device) < stack_p)
        stack = torch.where(extra, torch.randint(2, MAX_STACK + 1, (n,), device=self.device), stack)
        level = torch.zeros(n, dtype=torch.long, device=self.device)
        multi = stack > 1
        if multi.any():
            # pick a uniform level in [0, stack_n)
            rnd = torch.rand(n, device=self.device)
            level = torch.where(multi, (rnd * stack.float()).long().clamp(max=MAX_STACK - 1), level)
            level = torch.minimum(level, stack - 1)
        self.stack_n[env_ids] = stack
        self.target_level[env_ids] = level

        # body yaw ≈ π so forks point +X (into the pallet). Tips 1–2 m in front of the face.
        body_yaw = math.pi + sample_uniform(-1.0, 1.0, (n,), self.device) * yaw_hi
        tip_x = -sample_uniform(self.cfg.staging_tip_min, self.cfg.staging_tip_max, (n,), self.device)
        tip_y = sample_uniform(-1.0, 1.0, (n,), self.device) * lat_hi
        # root so that the *mean* tip lands at (tip_x, tip_y). Tips are ~2.138 m along forks
        # (body -X). At yaw=π, body -X is world +X, so root_x ≈ tip_x - 2.138.
        fork = torch.stack((torch.cos(body_yaw + math.pi), torch.sin(body_yaw + math.pi)), dim=-1)
        root_xy = torch.stack((tip_x, tip_y), dim=-1) - 2.138 * fork
        root_pose = self.robot.data.default_root_pose.torch[env_ids].clone()
        root_pose[:, :2] = root_xy + self.scene.env_origins[env_ids, :2]
        root_pose[:, 2] = 0.02
        root_pose[:, 3:7] = _yaw_quat(body_yaw)
        pocket_z = POCKET_Z_CENTER + level.float() * PALLET_HEIGHT
        # Rest-pose tip z is 0.221 with the USD root at z=0. Root is written at 0.02, so
        # tip_z ≈ 0.241 + lift. Ground pockets (z=0.065) need lift at the lower limit.
        lift0 = (pocket_z - 0.241 + sample_uniform(-1.0, 1.0, (n,), self.device) * lift_hi).clamp(*LIFT_LIMITS)

        joint_pos = self.robot.data.default_joint_pos.torch[env_ids].clone()
        joint_pos[:, self._steer_idx] = sample_uniform(-0.3, 0.3, (n, 1), self.device) * self._steer_scale
        joint_pos[:, self._lift_idx] = lift0.unsqueeze(-1)
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=self.device), env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)

        self._place_collision(env_ids)
        self.delayed[env_ids] = torch.rand(n, device=self.device) < self.cfg.action_delay_prob
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.applied_actions[env_ids] = 0.0

        self.prev_potential[env_ids] = self._potential()[env_ids]

    def _place_collision(self, env_ids: torch.Tensor):
        n = len(env_ids)
        origin = self.scene.env_origins[env_ids]
        z0 = self.target_level[env_ids].float() * PALLET_HEIGHT
        q = torch.zeros(n, 4, device=self.device)
        q[:, 3] = 1.0

        def write(asset, xy, z):
            pose = torch.zeros(n, 7, device=self.device)
            pose[:, 0] = origin[:, 0] + xy[0]
            pose[:, 1] = origin[:, 1] + xy[1]
            pose[:, 2] = z
            pose[:, 3:7] = q
            asset.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
            asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=self.device), env_ids=env_ids)

        write(self.face_left, (0.025, -0.55), z0 + 0.5 * PALLET_HEIGHT)
        write(self.face_center, (0.025, 0.0), z0 + 0.5 * PALLET_HEIGHT)
        write(self.face_right, (0.025, 0.55), z0 + 0.5 * PALLET_HEIGHT)

        unused = torch.full((n,), -2.0, device=self.device)
        z1 = torch.where(self.stack_n[env_ids] > 1, torch.full((n,), 1.5 * PALLET_HEIGHT, device=self.device), unused)
        z2 = torch.where(self.stack_n[env_ids] > 2, torch.full((n,), 2.5 * PALLET_HEIGHT, device=self.device), unused)
        # hide the extra deck that *is* the target (face boards occupy that level)
        z1 = torch.where(self.target_level[env_ids] == 1, unused, z1)
        z2 = torch.where(self.target_level[env_ids] == 2, unused, z2)
        write(self.stack_1, (0.40, 0.0), z1)
        write(self.stack_2, (0.40, 0.0), z2)

    # ------ debug visualization ------
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "insert_markers"):
                cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/forklift_insert",
                    markers={
                        "left": sim_utils.SphereCfg(
                            radius=0.04, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.9, 0.2))
                        ),
                        "right": sim_utils.SphereCfg(
                            radius=0.04, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.6, 1.0))
                        ),
                    },
                )
                self.insert_markers = VisualizationMarkers(cfg)
            self.insert_markers.set_visibility(True)
        elif hasattr(self, "insert_markers"):
            self.insert_markers.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not hasattr(self, "insert_markers"):
            return
        tgt_l, tgt_r = self._pocket_targets()
        pos = torch.cat((tgt_l, tgt_r), dim=0)
        quat = torch.zeros(pos.shape[0], 4, device=self.device)
        quat[:, 3] = 1.0
        idx = torch.cat(
            (torch.zeros(self.num_envs, dtype=torch.long, device=self.device), torch.ones(self.num_envs, dtype=torch.long, device=self.device))
        )
        self.insert_markers.visualize(pos, quat, marker_indices=idx)
