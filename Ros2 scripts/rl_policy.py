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
