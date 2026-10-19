"""Closed-loop check of the deployed insert policy (numpy MLP + the env's 3x64 scan).

    ~/IsaacLab/isaaclab.sh -p scripts/eval_insert.py \\
        [--policy ../Ros2\\ scripts/policies/forklift_insert.npz] [--num_envs 64] [--seconds 90]
"""

import argparse
import sys
from pathlib import Path

from isaaclab.app import launch_simulation
from isaaclab.cli import load_external_tasks

from isaaclab_rl.entrypoints.backends import play_rsl_rl
from isaaclab_rl.entrypoints.common import create_isaaclab_env
from isaaclab_tasks.utils import resolve_task_config

HERE = Path(__file__).resolve().parents[1]
ROS_DIR = HERE.parent / "Ros2 scripts"
sys.path.insert(0, str(ROS_DIR))
from rl_policy import NumpyPolicy, build_insert_obs  # noqa: E402

TASK = "Isaac-Forklift-Insert-Direct"
LIFT_LO, LIFT_HI = -0.15, 2.0


def main():
    load_external_tasks()
    own = argparse.ArgumentParser(add_help=False)
    own.add_argument("--policy", default=str(ROS_DIR / "policies" / "forklift_insert.npz"))
    own.add_argument("--seconds", type=float, default=90.0)
    own.add_argument("--level", type=float, default=1.0)
    own.add_argument("--range_noise", type=float, default=0.0)
    own_args, rest = own.parse_known_args(sys.argv[1:])
    if "--task" not in rest:
        rest += ["--task", TASK]
    if "--num_envs" not in rest:
        rest += ["--num_envs", "64"]
    args_cli = play_rsl_rl._parse_args(rest)

    env_cfg, _ = resolve_task_config(args_cli.task, args_cli.agent, play_mode=True)
    with launch_simulation(env_cfg, args_cli):
        import numpy as np  # noqa: PLC0415
        import torch  # noqa: PLC0415

        play_rsl_rl.apply_env_overrides(args_cli, env_cfg)
        env_cfg.curriculum_success_threshold = 2.0
        env = create_isaaclab_env(args_cli.task, env_cfg, args_cli, convert_marl_to_single_agent=False)
        fl = env.unwrapped
        fl.curriculum_level = own_args.level
        env.reset()

        policy = NumpyPolicy(own_args.policy)
        meta = policy.meta
        n = fl.num_envs
        last_action = np.zeros((n, 3))
        steps_in_episode = np.zeros(n, dtype=int)
        results = []

        for _ in range(int(own_args.seconds / fl.step_dt)):
            ranges = fl._lidar_ranges().cpu().numpy()
            if own_args.range_noise > 0.0:
                ranges = np.clip(ranges + own_args.range_noise * np.random.randn(*ranges.shape), 0.0, meta["lidar_max_range"])
            lift = fl.robot.data.joint_pos.torch[:, fl._lift_idx[0]].cpu().numpy()
            lift_norm = (lift - LIFT_LO) / (LIFT_HI - LIFT_LO) * 2.0 - 1.0
            actions = np.zeros((n, 3))
            for i in range(n):
                actions[i] = policy(build_insert_obs(meta, ranges[i], last_action[i], lift_norm[i]))
            last_action = actions

            lp, rp, _, _ = fl._tips_in_pallet()
            _, _, terminated, truncated, _ = env.step(torch.tensor(actions, dtype=torch.float32, device=fl.device))
            steps_in_episode += 1
            done = (terminated | truncated).cpu().numpy()
            reached = fl.reached.cpu().numpy()
            failed = fl.failed.cpu().numpy()
            face = fl.face_hit.cpu().numpy()
            for i in np.nonzero(done)[0]:
                if reached[i]:
                    outcome = "reached"
                elif face[i]:
                    outcome = "face"
                elif failed[i]:
                    outcome = "failed"
                else:
                    outcome = "timeout"
                depth = 0.5 * (float(lp[i, 0]) + float(rp[i, 0]))
                lat = 0.5 * (float(lp[i, 1]) + float(rp[i, 1]))
                results.append(
                    (outcome, steps_in_episode[i] * fl.step_dt, int(fl.stack_n[i]), int(fl.target_level[i]), depth, lat)
                )
                steps_in_episode[i] = 0
                last_action[i] = 0.0

        print("\n================ deployed insert-policy closed-loop check ================")
        print(f"policy {own_args.policy}\n({meta.get('checkpoint')}), level {own_args.level}")
        if not results:
            print("no finished episodes")
        else:
            ok = [r for r in results if r[0] == "reached"]
            face = sum(r[0] == "face" for r in results)
            fail = sum(r[0] == "failed" for r in results)
            t = np.mean([r[1] for r in ok]) if ok else float("nan")
            print(
                f"episodes {len(results):5d}  reached {100 * len(ok) / len(results):5.1f}%  "
                f"face-hit {100 * face / len(results):4.1f}%  failed {100 * fail / len(results):4.1f}%  "
                f"timeout {100 * (len(results) - len(ok) - face - fail) / len(results):4.1f}%  "
                f"mean time {t:5.1f} s"
            )
            for stacks in sorted({r[2] for r in results}):
                rs = [r for r in results if r[2] == stacks]
                rok = sum(r[0] == "reached" for r in rs)
                print(f"  stack_n={stacks}: {rok}/{len(rs)} reached")
            to = [r for r in results if r[0] == "timeout"]
            if to:
                print(
                    f"  timeouts: mean depth {np.mean([r[4] for r in to]):.2f} m  "
                    f"mean |lat| {np.mean([abs(r[5]) for r in to]):.2f} m"
                )
        print("=========================================================================\n")
        env.close()


if __name__ == "__main__":
    main()
