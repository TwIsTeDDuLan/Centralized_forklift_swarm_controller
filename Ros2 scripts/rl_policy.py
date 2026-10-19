"""
ROS-free part of the RL forklift controller: the exported actor MLP and its observation vector.

Shared by rl_forklift_controller.py (ROS node) and isaac_lab_forklift/scripts/eval_deploy.py (closed-loop
check in Isaac Lab), so the deployed observation pipeline is the one that gets tested.
Only needs numpy.
"""

import json
import math

import numpy as np


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class NumpyPolicy:
    """Deterministic actor: MLP with the activation used in training, output = action mean."""

    def __init__(self, path: str):
        data = np.load(path)
        self.meta = json.loads(str(data['meta_json']))
        n = sum(1 for k in data.files if k.startswith('W'))
        self.layers = [(data[f'W{i}'].astype(np.float64), data[f'b{i}'].astype(np.float64)) for i in range(n)]
        act = self.meta['activation']
        if act != 'elu':
            raise ValueError(f"unsupported activation '{act}'")

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = obs
        for i, (w, b) in enumerate(self.layers):
            x = w @ x + b
            if i < len(self.layers) - 1:
                x = np.where(x > 0.0, x, np.expm1(np.minimum(x, 0.0)))  # elu
        return np.clip(x, -1.0, 1.0)


def build_actor_obs(meta, ref_xy, body_yaw, goal_xy, goal_yaw, last_action):
    """Same layout as ForkliftNavEnv._get_observations (actor). All positions in meters, y up."""
    dx, dy = goal_xy[0] - ref_xy[0], goal_xy[1] - ref_xy[1]
    c, s = math.cos(body_yaw), math.sin(body_yaw)
    gx, gy = c * dx + s * dy, -s * dx + c * dy
    dist = math.hypot(dx, dy)
    scale, clip = meta['pos_obs_scale'], meta['pos_obs_clip']
    far = max(dist * scale / clip, 1.0)
    if goal_yaw is None:
        sin_e = cos_e = has = 0.0
    else:
        err = wrap(goal_yaw - body_yaw)
        sin_e, cos_e, has = math.sin(err), math.cos(err), 1.0
    return np.array([gx * scale / far, gy * scale / far, min(dist * scale, clip),
                     sin_e, cos_e, has, last_action[0], last_action[1]])


def flatten_lidar_scan(ranges, max_range):
    """Normalize a 3x64 (or flat 192) range image to [0, 1], same order as ForkliftInsertEnv."""
    r = np.asarray(ranges, dtype=np.float64).reshape(-1)
    return np.clip(r, 0.0, max_range) / max_range


def staging_tag_pixels(pallet_xy, fork_heading_rad, px_per_m, tip_standoff=1.5, tag_to_tip=2.518, face_offset=0.4):
    """Tag pixel pose so the fork tips sit `tip_standoff` m in front of the pallet's entry face.

    Image frame: x right, y down. `fork_heading_rad` is the tracker convention (0 right, 90 up).
    `pallet_xy` is the pallet tag (deck centre); the 1200 mm entry face is `face_offset` (half the
    800 mm depth) in front of it.
    """
    dist_px = (face_offset + tip_standoff + tag_to_tip) * px_per_m
    ux, uy = math.cos(fork_heading_rad), -math.sin(fork_heading_rad)
    return pallet_xy[0] - dist_px * ux, pallet_xy[1] - dist_px * uy


def pallet_entry_heading(pallet_theta, forklift_xy, pallet_xy):
    """Fork heading [rad, tracker convention] that enters a 1200 mm face, picking the face toward the forklift.

    The pallet tag's top edge points along the normal of a 1200 mm face (measured in Environment.usd),
    so the forks must point along +-theta; the 800 mm faces do not fit the tines.
    """
    to_pallet = math.atan2(-(pallet_xy[1] - forklift_xy[1]), pallet_xy[0] - forklift_xy[0])
    return pallet_theta if math.cos(pallet_theta - to_pallet) >= 0.0 else wrap(pallet_theta + math.pi)


def lift_joint_norm(height_m, lo=-0.15, hi=2.0):
    return max(-1.0, min(1.0, (height_m - lo) / (hi - lo) * 2.0 - 1.0))


def build_insert_obs(meta, ranges, last_action, lift_norm):
    """Same layout as ForkliftInsertEnv._get_observations (actor).

    ranges: 192 floats (3 elevation channels x 64 azimuth, channel-major), metres.
    last_action: (speed, steer, lift) in [-1, 1].
    lift_norm: lift joint mapped to [-1, 1] using the USD limits (-0.15, 2.0).
    """
    lidar = flatten_lidar_scan(ranges, meta['lidar_max_range'])
    act = np.asarray(last_action, dtype=np.float64).reshape(-1)[:3]
    if act.shape[0] < 3:
        act = np.pad(act, (0, 3 - act.shape[0]))
    return np.concatenate([lidar, act, [float(lift_norm)]])
