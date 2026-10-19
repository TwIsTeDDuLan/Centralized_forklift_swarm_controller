# RL go-to-pose for forklift_b (Isaac Lab 3.0 + RSL-RL PPO)

Trains `Issasim Models/forklift_b.usd` (rear-wheel Ackermann steering) to drive its tracked ArUco tag to
a target `(x, y)`, optionally with a final fork heading, then runs the policy as a ROS 2 node with the
**same topics** as the hand-tuned `forklift_controller.py`. The fleet controller and task allocation
(EOITAG) can therefore switch between the two without changes.

```
isaac_lab_forklift/                 <- Local Disk: code, logs, checkpoints
  forklift_lab/robot.py             articulation cfg (USD gains, fork tips, lidar mount)
  forklift_lab/pallet.py            EUR pocket geometry + o3dyn visual spawn
  forklift_lab/tasks/               Isaac-Forklift-Nav-Direct, Isaac-Forklift-Insert-Direct
  scripts/train.sh | play.sh        wrappers around Isaac Lab's own train.py / play.py
  scripts/export_policy.py          checkpoint -> numpy weights for ROS (no sim needed)
  scripts/drive_check.py            physics / sign-convention check with fixed commands
  scripts/eval_deploy.py            closed-loop nav policy (emulated camera)
  scripts/eval_insert.py            closed-loop insert policy (env scan + numpy actor)
  logs/                             rsl_rl runs (git-ignored)
Ros2 scripts/
  rl_policy.py                      numpy MLP + nav/insert observation builders
  rl_forklift_controller.py         two-policy ROS 2 node (nav then insert)
  run_rl_forklift.sh                ./run_rl_forklift.sh 1 2 3
  policies/forklift_nav.npz, forklift_insert.npz
  attach_fork_lidar.py              warehouse 3x64 scan -> /forklift_N/scan
```

## Commands

```bash
cd "/media/twisted/Local Disk/Projects/Robotics/Centralized_forklift_swarm_controller/isaac_lab_forklift"
./scripts/train.sh                                    # nav, background, 1024 envs, 1000 iterations (~30 min)
TASK=Isaac-Forklift-Insert-Direct ./scripts/train.sh --num_envs 256
tail -f logs/train_Isaac-Forklift-Insert-Direct.log | grep -E --line-buffered 'Learning iteration|success_rate|face_hit'
~/IsaacLab/.venv/bin/tensorboard --logdir logs/rsl_rl
TASK=Isaac-Forklift-Insert-Direct ./scripts/play.sh
~/IsaacLab/.venv/bin/python scripts/export_policy.py --experiment forklift_insert
source ~/IsaacLab/.venv/bin/activate && ~/IsaacLab/isaaclab.sh -p scripts/eval_insert.py --level 1.0
```

Pallet pickup on the warehouse scene: launch the RL nodes, attach the fork lidar (Script Editor or
`isaaclab.sh -p ../Ros2\ scripts/attach_fork_lidar.py` with the scene open), then `pallet 1 1` in
the fleet controller. Staging success is 0.5 m / ~30°. Insert success is both tips 0.40 m into the
pockets (reward target 0.70 m) with no face-stringer contact. Run `eval_insert.py` before claiming
the insert policy works.

Then, with the Isaac Sim warehouse and `vision_tracker.py` running:

```bash
cd "../Ros2 scripts" && ./run_rl_forklift.sh 1 2 3
ros2 topic pub --once /forklift_1/lift_target_pos geometry_msgs/msg/Pose2D "{x: 1087.0, y: 360.0, theta: 30.0}"
```

## Current policies

Nav (`2026-10-18_11-46-17/model_999`, `forklift_nav.npz`). From `eval_deploy.py --level 1`: goals
1–9 m away, the deployed numpy code path, int-pixel vision with the heading low-pass, and 50% of
envs with one extra step of latency.

| Goals | Reached | Failed | Timeout (45 s) | Mean time |
|---|---|---|---|---|
| any heading (`theta = NaN`) | 100.0% | 0% | 0% | 9.5 s (0.66 m/s average) |
| with heading (±10° in training) | 92.7% | 0% | 7.3% | 16.2 s |

Insert (`2026-10-19_00-31-36/model_1500`, `forklift_insert.npz`). From `eval_insert.py` (env 3×64
scan + numpy actor):

| Curriculum | Reached | Face-hit | Failed | Timeout | Mean time |
|---|---|---|---|---|---|
| level 0 (tight staging, single deck) | 100.0% | 0% | 0% | 0% | 3.3 s |
| level 1 (max pose/lift noise, stacks) | 77.9% | 0% | 0% | 22.1% | 3.3 s |

Not yet tested: the ROS node driving the Isaac Sim warehouse scene (`Environment.usd`) with the real
camera, `vision_tracker.py`, and `attach_fork_lidar.py`.

## Changes from the original plan, and why

| Plan | Now | Why |
|---|---|---|
| Custom `train.py` / `play.py` | Isaac Lab's unified scripts; the task registers through the `isaaclab.tasks` entry point (like `q8bot_lab`) | Nothing to maintain across Isaac Lab updates. `play.py` already exports `policy.pt` + `.onnx`. |
| Symlink Isaac Lab's `logs/` to Local Disk | Run from `isaac_lab_forklift/`; Isaac Lab writes `logs/` relative to the working directory | No change to `~/IsaacLab`, so q8bot's runs stay where they are. Only a ~1 kB editable-install link is on the NVMe. |
| TorchScript in the ROS node | numpy weights (`.npz`) + 15-line MLP | The ROS (system) Python has numpy but no torch / onnxruntime. Checked: numpy output matches torch to 3e-7. |
| Obs: body velocity, wheel speed, steering angle | Actor gets only what ROS has: goal in body frame, distance, sin/cos heading error, has-heading flag, last action. The critic also gets the true velocities and joint states. | The camera gives tag x, y, theta and nothing else. Asymmetric actor-critic keeps training fast without a sim-only actor. |
| Raw wrapped heading error | sin / cos of the error (zeroed when no heading is asked for) | No jump at ±180°. |
| Pixels (implicit) | Train in meters. The node converts with `px_per_m` = 65 (camera 20 m up, tag 2.95 m high, 60° FOV) and flips image y. | The policy doesn't depend on the camera, so it can be reused for other robots and cameras. |
| Generic reward table | Potential-based progress (distance + heading error weighted near the goal), +20 for stopping inside tolerance, failure −10, small time / action / action-rate costs | Potential shaping can't create reward loops. Success requires being *stopped*, so the policy learns to stop on the spot. |
| Fixed goal range | Curriculum 3 m → 9 m by success rate | Short goals first; the camera view is ~19.7 × 11 m. |
| — | Randomized 0/1-step action latency, camera noise (2 cm, 0.03 rad); `eval_deploy.py` adds int-pixel truncation and the tracker's heading low-pass | Robustness to the ROS loop and the vision tracker. |
| — | 60 Hz physics, 10 Hz policy | Same rates as the Isaac Sim scene and the ROS controller's timer. |

## Conventions (checked with `drive_check.py`)

* Body +X = counterweight side; forks point along −X.
* `vision_tracker` theta (the tag's top edge) points along the **forks**: measured from the tag mesh UVs
  in `Environment.usd`. So the RL node uses `tag_heading_offset_deg = 0`. `forklift_controller.py`
  assumes 180, which its drive/steering sign auto-calibration hides for position goals, but heading
  goals then come out 180° off. The RL node also measures the offset from its own driving (in 90° steps)
  and corrects it (`auto_heading_check`).
* Goal `theta` [deg] = the direction the forks should point when the forklift stops, in the tracker's
  convention: 0 = image right, 90 = image up, 180 = image left, −90 = image down. NaN = any heading.
* Actions are joint targets in Ackermann-topic units: `speed = a0 * 240 deg/s`, `steering = a1 * 55 deg`.
  Measured in the sim: negative speed = forks first, and (+speed, +steer) yaws clockwise. These are the
  same signs as the hand-tuned controller's defaults (`drive_sign = -1`, `steering_sign = -1`).
  The policy learned them, so the RL node needs no sign calibration.
* Top speed 0.64 m/s (240 deg/s × 0.16 m); about 1.2 m turn radius at full lock.

## Sim-to-sim notes

* `forklift_b.usd` authors `stiffness = 100` on the continuous `back_wheel_drive` joint. With a velocity
  target this spring pulls the wheel back toward 0° as it winds up, so the drive slowly loses
  authority. Training sets it to 0. To make Isaac Sim match, set
  `back_wheel_joints/back_wheel_drive` → *Drive* → *Stiffness* = 0 in `Forklift_ROS2.usd`.
* Steering and mast gains come from the USD, so they match Isaac Sim.
* The ArUco tag position in the body frame (`TAG_OFFSET_B = (0.38, 0)`) was read from `Environment.usd`.

## Toward task allocation / the heterogeneous fleet

* Same goal and status topics (`lift_target_pos`, `goal_status`): the allocator can send goals without
  knowing which controller runs on which robot.
* Per-robot specifics (USD, joints, speed / steering limits, tag offset) live in `robot.py` and in the
  `.npz` metadata. To add a robot type (e.g. `forklift_c`), add a config and register a task.
* A useful cost for EOITAG: `eval_deploy.py` measures time-to-goal against distance and heading change
  for the trained policy. That table, or the critic's value, can stand in for travel cost when
  scoring forklift → task assignments.
