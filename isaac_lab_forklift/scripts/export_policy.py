"""Export a trained actor to plain numpy weights for the ROS node (the ROS Python has numpy but no torch).

No simulation is started; it only reads the checkpoint and the run's params/env.yaml, so the observation
and action scales stored next to the weights are exactly the ones the policy was trained with.

    ~/IsaacLab/.venv/bin/python scripts/export_policy.py
    ~/IsaacLab/.venv/bin/python scripts/export_policy.py --experiment forklift_insert
    ~/IsaacLab/.venv/bin/python scripts/export_policy.py --run 2026-10-18_12-00-00 --checkpoint model_800.pt

Writes <run>/exported/policy_numpy.npz and copies it to ../Ros2 scripts/policies/<experiment>.npz.
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parents[1]
ROS_DIR = HERE.parent / "Ros2 scripts" / "policies"

NAV_OBS = [
    "goal_x_b",
    "goal_y_b",
    "distance",
    "sin_heading_err",
    "cos_heading_err",
    "has_heading",
    "last_speed_action",
    "last_steer_action",
]
NAV_META = [
    "max_wheel_speed_deg",
    "max_steer_deg",
    "ref_offset_b",
    "pos_obs_scale",
    "pos_obs_clip",
    "pos_tolerance",
    "heading_tolerance",
    "stop_speed",
    "decimation",
]
INSERT_OBS = [f"lidar_{i}" for i in range(192)] + [
    "last_speed_action",
    "last_steer_action",
    "last_lift_action",
    "lift_joint_norm",
]
INSERT_META = [
    "max_wheel_speed_deg",
    "max_steer_deg",
    "max_lift_speed",
    "lidar_channels",
    "lidar_azimuth",
    "lidar_max_range",
    "lidar_h_fov_deg",
    "lidar_v_fov_deg",
    "decimation",
    "insert_success_x",
]


def latest(paths, key):
    paths = list(paths)
    if not paths:
        raise SystemExit("nothing found")
    return max(paths, key=key)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", default="forklift_nav", choices=("forklift_nav", "forklift_insert"))
    parser.add_argument("--run", help="run directory name under logs/rsl_rl/<experiment> (default: latest)")
    parser.add_argument("--checkpoint", help="checkpoint file name, e.g. model_800.pt (default: latest)")
    parser.add_argument("--out", type=Path, help="where the ROS node looks for the policy")
    args = parser.parse_args()

    log_root = HERE / "logs" / "rsl_rl" / args.experiment
    default_out = ROS_DIR / f"{args.experiment}.npz"
    out = args.out or default_out
    obs_layout = NAV_OBS if args.experiment == "forklift_nav" else INSERT_OBS
    meta_keys = NAV_META if args.experiment == "forklift_nav" else INSERT_META
    action_dim = 2 if args.experiment == "forklift_nav" else 3

    run = log_root / args.run if args.run else latest(log_root.iterdir(), key=lambda p: p.name)
    ckpt = run / args.checkpoint if args.checkpoint else latest(
        run.glob("model_*.pt"), key=lambda p: int(re.findall(r"\d+", p.name)[-1])
    )
    print(f"[export] checkpoint: {ckpt}")

    state = torch.load(ckpt, map_location="cpu", weights_only=False)["actor_state_dict"]
    if any("normalizer" in k for k in state):
        raise SystemExit("actor uses observation normalization; the numpy export expects it disabled")
    weights = sorted(
        (k for k, v in state.items() if k.endswith(".weight") and v.ndim == 2),
        key=lambda k: int(k.split(".")[-2]),
    )
    arrays = {}
    for i, wk in enumerate(weights):
        arrays[f"W{i}"] = state[wk].numpy().astype(np.float32)
        arrays[f"b{i}"] = state[wk[: -len("weight")] + "bias"].numpy().astype(np.float32)
    shapes = [arrays[f"W{i}"].shape for i in range(len(weights))]
    print(f"[export] layers: {shapes}")
    if shapes[0][1] != len(obs_layout) or shapes[-1][0] != action_dim:
        raise SystemExit(f"unexpected actor shape {shapes}; expected in={len(obs_layout)} out={action_dim}")

    env_cfg = yaml.load((run / "params" / "env.yaml").read_text(), Loader=yaml.UnsafeLoader)
    agent_cfg = yaml.load((run / "params" / "agent.yaml").read_text(), Loader=yaml.UnsafeLoader)
    meta = {k: env_cfg[k] for k in meta_keys}
    if "ref_offset_b" in meta:
        meta["ref_offset_b"] = list(meta["ref_offset_b"])
    meta["control_dt"] = env_cfg["sim"]["dt"] * env_cfg["decimation"]
    meta["activation"] = agent_cfg["actor"]["activation"]
    meta["obs_layout"] = obs_layout
    meta["checkpoint"] = str(ckpt.relative_to(HERE))
    meta["experiment"] = args.experiment
    arrays["meta_json"] = np.array(json.dumps(meta))

    out_run = run / "exported" / "policy_numpy.npz"
    out_run.parent.mkdir(exist_ok=True)
    np.savez(out_run, **arrays)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(out_run, out)
    print(f"[export] wrote {out_run}\n[export] copied to {out}")
    print(f"[export] meta: {json.dumps(meta, indent=1)}")


if __name__ == "__main__":
    main()
