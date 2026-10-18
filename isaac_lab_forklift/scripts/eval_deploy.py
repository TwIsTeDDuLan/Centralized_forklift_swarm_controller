"""Closed-loop check of the *deployed* policy path in Isaac Lab.

Drives the forklifts with the exported numpy policy through the same code the ROS node runs
(``Ros2 scripts/rl_policy.py``), fed by an emulated vision_tracker.py:
  * tag position in integer pixels (the tracker truncates to int), 65 px/m, image y down,
  * tag heading through the tracker's low-pass filter (alpha 0.3 per frame).
Goals come from the env (same distribution as training at curriculum level --level).

    ~/IsaacLab/isaaclab.sh -p scripts/eval_deploy.py [--policy ../Ros2\\ scripts/policies/forklift_nav.npz]
        [--num_envs 256] [--seconds 120] [--level 1.0] [--heading_alpha 0.3]
"""

import argparse
import math
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
from rl_policy import NumpyPolicy, build_actor_obs, wrap  # noqa: E402

TASK = "Isaac-Forklift-Nav-Direct"


def main():
    load_external_tasks()
    own = argparse.ArgumentParser(add_help=False)
    own.add_argument("--policy", default=str(ROS_DIR / "policies" / "forklift_nav.npz"))
    own.add_argument("--seconds", type=float, default=120.0)
    own.add_argument("--level", type=float, default=1.0, help="goal-distance curriculum level 0..1")
    own.add_argument("--px_per_m", type=float, default=65.0)
    own.add_argument("--heading_alpha", type=float, default=0.3, help="vision_tracker low-pass, 1 = off")
    own_args, rest = own.parse_known_args(sys.argv[1:])
    if "--task" not in rest:
        rest += ["--task", TASK]
    if "--num_envs" not in rest:
        rest += ["--num_envs", "256"]
    args_cli = play_rsl_rl._parse_args(rest)

    env_cfg, _ = resolve_task_config(args_cli.task, args_cli.agent, play_mode=True)
    with launch_simulation(env_cfg, args_cli):
        import numpy as np  # noqa: PLC0415
        import torch  # noqa: PLC0415

        play_rsl_rl.apply_env_overrides(args_cli, env_cfg)
        env_cfg.curriculum_success_threshold = 2.0  # freeze the curriculum at --level
        env = create_isaaclab_env(args_cli.task, env_cfg, args_cli, convert_marl_to_single_agent=False)
        fl = env.unwrapped
        fl.curriculum_level = own_args.level
        env.reset()

        policy = NumpyPolicy(own_args.policy)
        meta = policy.meta
        s = own_args.px_per_m
        n = fl.num_envs
        last_action = np.zeros((n, 2))
        vis_yaw = fl.robot.data.heading_w.torch.cpu().numpy().copy()
        steps_in_episode = np.zeros(n, dtype=int)
        results = []  # (outcome, seconds, start distance, had heading)

        for _ in range(int(own_args.seconds / fl.step_dt)):
            ref, yaw = fl._pose_2d()
            ref = ref.cpu().numpy()
            yaw = yaw.cpu().numpy()
            goal = fl.goal_pos.cpu().numpy()
            goal_yaw = fl.goal_yaw.cpu().numpy()
            has_heading = fl.has_heading.cpu().numpy()

            actions = np.zeros((n, 2))
            for i in range(n):
                # --- what vision_tracker.py publishes ---
                d = wrap(yaw[i] - vis_yaw[i])
                vis_yaw[i] = wrap(vis_yaw[i] + own_args.heading_alpha * d)
                u, v = int(ref[i, 0] * s + 5000), int(-ref[i, 1] * s + 5000)  # +5000: keep pixels positive
                gu, gv = goal[i, 0] * s + 5000, -goal[i, 1] * s + 5000
                # --- what rl_forklift_controller.py does with it (heading_offset 180 deg) ---
                body_yaw = wrap(vis_yaw[i] + math.pi - math.pi)
                g_yaw = wrap(goal_yaw[i]) if has_heading[i] > 0.5 else None
                obs = build_actor_obs(meta, (u / s, -v / s), body_yaw, (gu / s, -gv / s), g_yaw, last_action[i])
                actions[i] = policy(obs)
            last_action = actions

            _, _, terminated, truncated, _ = env.step(torch.tensor(actions, dtype=torch.float32, device=fl.device))
            steps_in_episode += 1
            done = (terminated | truncated).cpu().numpy()
            reached = fl.reached.cpu().numpy()
            failed = fl.failed.cpu().numpy()
            start = fl.start_dist.cpu().numpy()
            for i in np.nonzero(done)[0]:
                outcome = "reached" if reached[i] else ("failed" if failed[i] else "timeout")
                results.append((outcome, steps_in_episode[i] * fl.step_dt, start[i], has_heading[i] > 0.5))
                steps_in_episode[i] = 0
                last_action[i] = 0.0
            if done.any():
                new_yaw = fl.robot.data.heading_w.torch.cpu().numpy()
                vis_yaw[done] = new_yaw[done]

        print("\n================ deployed-policy closed-loop check ================")
        print(f"policy {own_args.policy}\n({meta['checkpoint']}), level {own_args.level}, "
              f"heading low-pass {own_args.heading_alpha}, int pixels at {s} px/m")
        for label, sel in (("all goals", lambda r: True), ("with heading", lambda r: r[3]),
                           ("any heading", lambda r: not r[3])):
            rs = [r for r in results if sel(r)]
            if not rs:
                continue
            ok = [r for r in rs if r[0] == "reached"]
            fail = sum(r[0] == "failed" for r in rs)
            t = np.mean([r[1] for r in ok]) if ok else float("nan")
            v = np.mean([r[2] / r[1] for r in ok]) if ok else float("nan")
            print(f"{label:13s} episodes {len(rs):5d}  reached {100 * len(ok) / len(rs):5.1f}%  "
                  f"failed {100 * fail / len(rs):4.1f}%  timeout {100 * (len(rs) - len(ok) - fail) / len(rs):4.1f}%  "
                  f"mean time {t:5.1f} s  mean (start dist / time) {v:4.2f} m/s")
        print("===================================================================\n")
        env.close()


if __name__ == "__main__":
    main()
