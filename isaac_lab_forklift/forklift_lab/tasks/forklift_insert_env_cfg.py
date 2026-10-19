"""Precision fork-insertion task: from a 1–2 m staging pose, align and slide into pallet pockets."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass, replace
from isaaclab_physx.physics import PhysxCfg

from ..pallet import PalletSceneAssetsCfg
from ..robot import FORKLIFT_B_CFG, FORK_TIP_LEFT_L, FORK_TIP_RIGHT_L, LIDAR_OFFSET_L, WHEEL_RADIUS


@configclass
class ForkliftInsertSceneCfg(PalletSceneAssetsCfg, InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0)
        ),
    )
    robot: ArticulationCfg = replace(
        FORKLIFT_B_CFG,
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=replace(FORKLIFT_B_CFG.spawn, activate_contact_sensors=True),
    )
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))
    # Lift body vs the three entry-face stringers. Pocket-wall rub is not a face hit (see env).
    contact_lift = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/lift",
        update_period=0.0,
        history_length=1,
        track_air_time=False,
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/FaceLeft",
            "{ENV_REGEX_NS}/FaceCenter",
            "{ENV_REGEX_NS}/FaceRight",
        ],
    )


@configclass
class ForkliftInsertEnvCfg(DirectRLEnvCfg):
    decimation = 6
    episode_length_s = 20.0
    sim: SimulationCfg = SimulationCfg(dt=1 / 60, render_interval=decimation, physics=PhysxCfg())
    scene: ForkliftInsertSceneCfg = ForkliftInsertSceneCfg(num_envs=256, env_spacing=5.0, replicate_physics=True)

    # ------ spaces ------
    # Actor: lidar (3*64) + last action (3) + lift joint (1)
    lidar_channels = 3
    lidar_azimuth = 64
    observation_space = 3 * 64 + 3 + 1  # 196
    # Critic: actor + left/right tip-to-pocket (6) + depth/lat/yaw/height err (4) + contact (1)
    state_space = 196 + 6 + 4 + 1
    action_space = 3

    # ------ actions ------
    max_wheel_speed_deg = 240.0
    max_steer_deg = 55.0
    max_lift_speed = 0.30  # [m/s]
    wheel_radius = WHEEL_RADIUS
    action_delay_prob = 0.3

    # ------ lidar (fork-mounted, lift +Y = forks) ------
    lidar_offset_l = LIDAR_OFFSET_L
    lidar_h_fov_deg = 90.0
    lidar_v_fov_deg = 16.0  # ±8 deg, 3 channels
    lidar_max_range = 4.0
    lidar_range_noise = 0.01  # [m]

    # ------ fork / pallet (must match pallet.py + robot.py) ------
    fork_tip_left_l = FORK_TIP_LEFT_L
    fork_tip_right_l = FORK_TIP_RIGHT_L

    # ------ reset / curriculum ------
    staging_tip_min = 1.0
    staging_tip_max = 2.0
    staging_lat_start = 0.08
    staging_lat_end = 0.25
    staging_yaw_start = 0.10  # [rad]
    staging_yaw_end = 0.35
    lift_err_start = 0.02
    lift_err_end = 0.08
    stack_prob_start = 0.0
    stack_prob_end = 0.45
    curriculum_success_threshold = 0.88
    curriculum_step = 0.004

    # ------ success / failure ------
    insert_success_x = 0.40
    pocket_y_tol = 0.20
    pocket_z_tol = 0.05
    face_force_threshold = 15.0  # [N]
    rew_face_hit = -50.0
    rew_progress = 10.0
    rew_success = 20.0
    rew_failure = -10.0
    rew_time = -0.02
    rew_action = -0.01
    rew_action_rate = -0.02

    debug_vis = True
