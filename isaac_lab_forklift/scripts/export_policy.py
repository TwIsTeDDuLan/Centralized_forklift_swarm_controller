"""Export a trained actor to plain numpy weights for the ROS node (the ROS Python has numpy but no torch).

No simulation is started; it only reads the checkpoint and the run's params/env.yaml, so the observation
and action scales stored next to the weights are exactly the ones the policy was trained with.

    ~/IsaacLab/.venv/bin/python scripts/export_policy.py                      # latest run, latest checkpoint
    ~/IsaacLab/.venv/bin/python scripts/export_policy.py --run 2026-10-18_12-00-00 --checkpoint model_800.pt

Writes <run>/exported/policy_numpy.npz and copies it to ../Ros2 scripts/policies/forklift_nav.npz.
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
LOG_ROOT = HERE / "logs" / "rsl_rl" / "forklift_nav"
ROS_POLICY = HERE.parent / "Ros2 scripts" / "policies" / "forklift_nav.npz"

# Must match ForkliftNavEnv._get_observations (actor part) and rl_forklift_controller.build_actor_obs.
OBS_LAYOUT = ["goal_x_b", "goal_y_b", "distance", "sin_heading_err", "cos_heading_err", "has_heading",
              "last_speed_action", "last_steer_action"]
META_KEYS = ["max_wheel_speed_deg", "max_steer_deg", "ref_offset_b", "pos_obs_scale", "pos_obs_clip",
             "pos_tolerance", "heading_tolerance", "stop_speed", "decimation"]


def latest(paths, key):
    paths = list(paths)
    if not paths:
        raise SystemExit("nothing found")
    return max(paths, key=key)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", help="run directory name under logs/rsl_rl/forklift_nav (default: latest)")
    parser.add_argument("--checkpoint", help="checkpoint file name, e.g. model_800.pt (default: latest)")
    parser.add_argument("--out", type=Path, default=ROS_POLICY, help="where the ROS node looks for the policy")
    args = parser.parse_args()

    run = LOG_ROOT / args.run if args.run else latest(LOG_ROOT.iterdir(), key=lambda p: p.name)
    ckpt = run / args.checkpoint if args.checkpoint else latest(
        run.glob("model_*.pt"), key=lambda p: int(re.findall(r"\d+", p.name)[-1])
    )
    print(f"[export] checkpoint: {ckpt}")

    state = torch.load(ckpt, map_location="cpu", weights_only=False)["actor_state_dict"]
    if any("normalizer" in k for k in state):
        raise SystemExit("actor uses observation normalization; the numpy export expects it disabled")
    # Linear layers in forward order: 2D '<...>.<index>.weight' tensors sorted by layer index.
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
    if shapes[0][1] != len(OBS_LAYOUT) or shapes[-1][0] != 2:
        raise SystemExit(f"unexpected actor shape {shapes}; update OBS_LAYOUT to match the env")

    env_cfg = yaml.load((run / "params" / "env.yaml").read_text(), Loader=yaml.UnsafeLoader)
    agent_cfg = yaml.load((run / "params" / "agent.yaml").read_text(), Loader=yaml.UnsafeLoader)
    meta = {k: env_cfg[k] for k in META_KEYS}
    meta["ref_offset_b"] = list(meta["ref_offset_b"])
    meta["control_dt"] = env_cfg["sim"]["dt"] * env_cfg["decimation"]
    meta["activation"] = agent_cfg["actor"]["activation"]
    meta["obs_layout"] = OBS_LAYOUT
    meta["checkpoint"] = str(ckpt.relative_to(HERE))
    arrays["meta_json"] = np.array(json.dumps(meta))

    out_run = run / "exported" / "policy_numpy.npz"
    out_run.parent.mkdir(exist_ok=True)
    np.savez(out_run, **arrays)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(out_run, args.out)
    print(f"[export] wrote {out_run}\n[export] copied to {args.out}")
    print(f"[export] meta: {json.dumps(meta, indent=1)}")


if __name__ == "__main__":
    main()
