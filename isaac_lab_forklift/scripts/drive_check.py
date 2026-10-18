"""Physics / sign-convention check: drive forklift_b with fixed Ackermann commands and report what happens.

Each group of envs holds one (speed, steering) command, in the same units as /forklift_N/ackermann.
Prints the body-frame travel direction, speed, yaw rate and turn radius, to compare against the
kinematic model and against what the hand-tuned ROS controller assumes (drive_sign, steering_sign).

    ./scripts/drive_check.sh            # or, from isaac_lab_forklift/:
    ~/IsaacLab/isaaclab.sh -p scripts/drive_check.py [--seconds 6]
"""

import argparse
import math
import sys

from isaaclab.app import launch_simulation
from isaaclab.cli import load_external_tasks

from isaaclab_rl.entrypoints.backends import play_rsl_rl
from isaaclab_rl.entrypoints.common import create_isaaclab_env
from isaaclab_tasks.utils import resolve_task_config

TASK = "Isaac-Forklift-Nav-Direct"
GROUPS = [  # name, (speed deg/s, steering deg)
    ("speed +240, steer 0", (240.0, 0.0)),
    ("speed -240, steer 0", (-240.0, 0.0)),
    ("speed +240, steer +55", (240.0, 55.0)),
    ("speed +240, steer -55", (240.0, -55.0)),
    ("speed -240, steer +55", (-240.0, 55.0)),
    ("speed +120, steer +30", (120.0, 30.0)),
]


def main():
    load_external_tasks()
    own = argparse.ArgumentParser(add_help=False)
    own.add_argument("--seconds", type=float, default=6.0)
    own_args, rest = own.parse_known_args(sys.argv[1:])
    if "--task" not in rest:
        rest += ["--task", TASK]
    if "--num_envs" not in rest:
        rest += ["--num_envs", str(len(GROUPS))]
    args_cli = play_rsl_rl._parse_args(rest)

    env_cfg, _ = resolve_task_config(args_cli.task, args_cli.agent, play_mode=True)
    with launch_simulation(env_cfg, args_cli):
        import torch  # noqa: PLC0415

        play_rsl_rl.apply_env_overrides(args_cli, env_cfg)
        env_cfg.episode_length_s = own_args.seconds + 10.0
        env_cfg.action_delay_prob = 0.0
        env_cfg.max_overshoot = 1.0e6
        env_cfg.goal_dist_min = env_cfg.goal_dist_max_start = 50.0  # never "reach" a goal
        env = create_isaaclab_env(args_cli.task, env_cfg, args_cli, convert_marl_to_single_agent=False)
        fl = env.unwrapped
        fl.reset()
        n = fl.num_envs
        cmd = torch.zeros(n, 2, device=fl.device)
        for i in range(n):
            speed, steer = GROUPS[i % len(GROUPS)][1]
            cmd[i, 0] = speed / env_cfg.max_wheel_speed_deg
            cmd[i, 1] = steer / env_cfg.max_steer_deg

        robot = fl.robot
        steps = int(own_args.seconds / fl.step_dt)
        settle = int(1.0 / fl.step_dt)  # let steering swing and speed build up first
        p0 = yaw0 = None
        for k in range(steps):
            fl.step(cmd)
            if k == settle:
                p0 = robot.data.root_pos_w.torch[:, :2].clone()
                yaw0 = robot.data.heading_w.torch.clone()
                yaw_unwrapped = torch.zeros_like(yaw0)
                yaw_prev = yaw0.clone()
            elif k > settle:
                yaw = robot.data.heading_w.torch
                yaw_unwrapped += torch.atan2(torch.sin(yaw - yaw_prev), torch.cos(yaw - yaw_prev))
                yaw_prev = yaw.clone()
        p1 = robot.data.root_pos_w.torch[:, :2]
        t = (steps - 1 - settle) * fl.step_dt
        tilt = robot.data.projected_gravity_b.torch[:, 2]

        print("\n================ forklift_b drive check ================")
        print(f"(kinematic model: 240 deg/s * r 0.16 m = {math.radians(240) * 0.16:.2f} m/s,"
              f" 55 deg steer -> R ~ {1.39 / math.tan(math.radians(55)):.2f} m at the rollers)")
        for i in range(n):
            d = p1[i] - p0[i]
            c, s = math.cos(yaw0[i].item()), math.sin(yaw0[i].item())
            bx, by = c * d[0].item() + s * d[1].item(), -s * d[0].item() + c * d[1].item()
            dist = math.hypot(bx, by)
            yaw_rate = yaw_unwrapped[i].item() / t
            radius = abs((dist / t) / yaw_rate) if abs(yaw_rate) > 1e-3 else float("inf")
            direction = "+X (counterweight first)" if bx > 0 else "-X (forks first)"
            print(f"{GROUPS[i % len(GROUPS)][0]:24s} start dir {direction:25s} "
                  f"chord speed {dist / t:5.2f} m/s  yaw rate {math.degrees(yaw_rate):+7.1f} deg/s  "
                  f"R {radius:5.2f} m  gravity_z {tilt[i].item():+.2f}")
        print("========================================================\n")
        env.close()


if __name__ == "__main__":
    main()
