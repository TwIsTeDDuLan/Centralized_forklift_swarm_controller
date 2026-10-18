"""Go-to-pose task for forklift_b: drive the tracked tag point to (x, y), optionally with a final heading."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass, replace
from isaaclab_physx.physics import PhysxCfg

from ..robot import FORKLIFT_B_CFG, TAG_OFFSET_B, WHEEL_RADIUS


@configclass
class ForkliftNavSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0)
        ),
    )
    robot: ArticulationCfg = replace(FORKLIFT_B_CFG, prim_path="{ENV_REGEX_NS}/Robot")
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))


@configclass
class ForkliftNavEnvCfg(DirectRLEnvCfg):
    # Isaac Sim runs the warehouse at 60 Hz and the ROS controller at 10 Hz: same rates here.
    decimation = 6
    episode_length_s = 45.0
    sim: SimulationCfg = SimulationCfg(dt=1 / 60, render_interval=decimation, physics=PhysxCfg())
    # Inter-env collisions are filtered, so envs may overlap while driving.
    scene: ForkliftNavSceneCfg = ForkliftNavSceneCfg(num_envs=1024, env_spacing=6.0, replicate_physics=True)

    # ------ spaces ------
    # Actor (deployable, only what the ROS node has: tag pose from the camera + its own last command):
    #   goal_b (2), distance (1), sin/cos heading error (2), has_heading (1), last action (2)
    observation_space = 8
    # Critic: actor obs + true body velocity (3), joint states (2), noise-free distance / heading error (2).
    state_space = 15
    action_space = 2

    # ------ actions (same units as the ROS Ackermann topic) ------
    max_wheel_speed_deg = 240.0  # [deg/s] back_wheel_drive -> 240 deg/s * 0.16 m = 0.67 m/s
    max_steer_deg = 55.0  # [deg] back_wheel_swivel (USD limit 60)
    wheel_radius = WHEEL_RADIUS
    # Probability that an env runs with one control step (0.1 s) of extra actuation latency (ROS round trip).
    action_delay_prob = 0.5

    # ------ observations ------
    ref_offset_b = TAG_OFFSET_B  # the point that must reach the goal = the ArUco tag the camera tracks
    pos_obs_scale = 0.2  # 1 / 5 m
    pos_obs_clip = 2.0  # goals further than 10 m look like 10 m
    pos_noise_std = 0.02  # [m] ~1.3 px at 65 px/m
    yaw_noise_std = 0.03  # [rad]

    # ------ goals / curriculum ------
    goal_dist_min = 1.0  # [m]
    goal_dist_max_start = 3.0  # [m] at curriculum level 0
    goal_dist_max_end = 9.0  # [m] at curriculum level 1 (camera view is ~19.7 x 11 m)
    heading_goal_prob = 0.75  # remainder: "any heading" goals (ROS theta = NaN)
    curriculum_success_threshold = 0.7
    curriculum_step = 0.01

    # ------ success / failure ------
    pos_tolerance = 0.2  # [m] (~13 px)
    heading_tolerance = 0.17  # [rad] (~10 deg)
    stop_speed = 0.1  # [m/s] must be (nearly) stopped to count as reached
    max_overshoot = 4.0  # [m] fail if this much further away than the initial distance

    # ------ rewards ------
    rew_progress = 10.0  # x decrease of the potential (distance + weighted heading error) per step
    heading_potential_weight = 1.0  # [m per rad] of heading error in the potential
    heading_potential_sigma = 1.5  # [m] heading error matters within ~this distance of the goal
    rew_success = 20.0
    rew_failure = -10.0
    rew_time = -0.02
    rew_action = -0.01
    rew_action_rate = -0.02

    debug_vis = True
