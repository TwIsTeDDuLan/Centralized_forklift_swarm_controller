# Agent guide: Centralized Forklift Swarm Controller

Isaac Sim warehouse with three rear-steered (Ackermann) forklifts, an overhead camera, and ROS 2 Jazzy nodes
that drive them. An RL go-to-pose policy, trained in Isaac Lab, can replace the hand-tuned driving controller.
The next goal is task allocation (Efficacy-Optimized Incremental Task Allocation Graph Search, EOITAG). That is
part of a larger heterogeneous multi-robot system with no fixed central controller: the coordinator assigns each
task to a suitable robot. Design every interface (topics, goal format, per-robot config) to stay robot-agnostic.

## Layout

```
Issasim Models/          USD assets: Environment.usd (warehouse, 3 forklifts, camera, pallets),
                         Forklift_ROS2.usd (forklift_b + ROS action graph), forklift_b.usd, forklift_c.usd, pallets
assets/ArUco Tags/       tag images (forklifts = ids 1-3, pallets = ids 4-6, DICT_4X4_50)
Ros2 scripts/            ROS 2 nodes (system Python 3.12, numpy only; no torch)
  vision_tracker.py        overhead camera -> /forklift_N/vision_data, pallet_N/vision_data
  forklift_controller.py   hand-tuned go-to-pose controller (base class)
  Fork Lifts/Fork_Lift_N.py  launch the hand-tuned controller for forklift N
  rl_forklift_controller.py  RL controller (subclass of ForkliftController, same topics)
  rl_policy.py             numpy MLP + observation builder (no ROS imports; shared with eval_deploy.py)
  policies/forklift_nav.npz  exported policy (weights + metadata)
  fleet_controller.py, run_*.sh, cmds.txt (example topic commands)
isaac_lab_forklift/      Isaac Lab package `forklift_lab` (task Isaac-Forklift-Nav-Direct); see its README.md
  forklift_lab/robot.py    articulation cfg, joint names, geometry constants
  forklift_lab/tasks/      DirectRLEnv + cfg + RSL-RL PPO cfg (agents/)
  scripts/                 train.sh, play.sh, export_policy.py, drive_check.py, eval_deploy.py
  logs/                    training runs (git-ignored)
```

## Machine and environment constraints

* **Disk:** `/` (NVMe, ext4) has under 10 GB free; don't put large outputs there. The project lives on
  `/media/twisted/Local Disk` (NTFS, ~110 GB free). Training logs and checkpoints go to
  `isaac_lab_forklift/logs/`, because Isaac Lab writes `logs/` relative to the working directory and the
  scripts `cd` there. Don't symlink or redirect `~/IsaacLab/logs`; another project (q8bot_lab) uses it.
* **Isaac Lab 3.0.0** at `~/IsaacLab` with its uv venv `~/IsaacLab/.venv`. Isaac Sim is at `~/issacsim`
  (symlinked as `~/IsaacLab/_isaac_sim`). GPU: RTX 3060 12 GB.
  * Launch with `source ~/IsaacLab/.venv/bin/activate && ~/IsaacLab/isaaclab.sh -p <script>`. Plain
    `.venv/bin/python` fails with "Isaac Sim is not installed".
  * There is no `--headless`: headless is the default. Use `--viz kit` for the viewer.
  * Quaternions are `(x, y, z, w)`. Asset data is `ProxyArray` (use `.torch`); e.g. `root_pos_w`,
    `heading_w`, `root_lin_vel_b`, `projected_gravity_b`.
  * Joint target setters (`set_joint_velocity_target_index`, ...) still work but are deprecated.
  * Through the gym wrapper, call `env.reset()`, not `env.unwrapped.reset()`, before `env.step()`.
  * Tasks are found through the `isaaclab.tasks` entry point in `isaac_lab_forklift/pyproject.toml`
    (editable-installed into the Isaac Lab venv with
    `uv pip install --python ~/IsaacLab/.venv/bin/python --no-deps --no-build-isolation -e isaac_lab_forklift`).
    Use Isaac Lab's own `scripts/reinforcement_learning/{train,play}.py`; don't copy them.
* **ROS 2 Jazzy** at `/opt/ros/jazzy`: `source /opt/ros/jazzy/setup.bash` before importing `rclpy`.
  For tests that must not disturb a running sim, use a separate `ROS_DOMAIN_ID`.
* Isaac Sim runs at 60 Hz; ROS controllers run at 10 Hz (0.1 s timer). Training uses the same rates
  (dt 1/60, decimation 6).

## Conventions (measured, don't re-guess)

* **forklift_b body frame:** +X = counterweight side, forks point along −X.
  * Driven + steered rear wheel: sphere, r = 0.16 m, at x = +0.58.
  * Passive rollers: x = −0.91 / −0.72. Wheelbase ≈ 1.39 m.
  * ArUco tag on the mast: body (0.38, 0), ~2.95 m high.
* **Joints:** `back_wheel_drive` (velocity), `back_wheel_swivel` (position, ±60°), `lift_joint`
  (prismatic, −0.15…2.0 m), `*_roller` (passive).
* **Ackermann topic = joint targets:** `/forklift_N/ackermann` `drive.speed` is the wheel velocity target
  in deg/s; `steering_angle` is the swivel target in deg (action graph in `Forklift_ROS2.usd`).
* **Signs** (`isaac_lab_forklift/scripts/drive_check.py`): negative speed drives forks-first,
  (+speed, +steer) yaws clockwise, 240 deg/s ≈ 0.64 m/s, ~1.2 m turn radius at 55°.
  These match `drive_sign = -1`, `steering_sign = -1` in `forklift_controller.py`.
* **Camera** (`Environment.usd` `ROS_cam`): 20 m up, looking straight down, 1280×720. Image x = world +X,
  image up = world +Y, **65 px/m at tag height**.
* **vision_data:** tag center in integer pixels (image y down). `theta` [rad] is CCW from image +x with
  image up = +90°; it's low-pass filtered (alpha 0.3).
  * **`theta` is the fork direction:** the tag's top edge points along the forks (measured from the tag
    mesh UVs). `rl_forklift_controller.py` therefore uses `tag_heading_offset_deg = 0`.
  * `forklift_controller.py` defaults `heading_offset_deg = 180`, which is wrong. Its sign auto-calibration
    hides this for position goals, but heading goals come out 180° off.
* **Goal** `/forklift_N/lift_target_pos` (Pose2D): `x, y` = where the **tag** should end, in pixels;
  `theta` = final **fork** heading in degrees (0 right, 90 up, 180 left, −90 down) or NaN = any heading.
  Goals outside a 50 px margin of the image are rejected.

## RL pipeline

* **Env** `forklift_lab/tasks/forklift_nav_env.py` (`ForkliftNavEnv`):
  * The tag point must reach a goal 1–9 m away, with a curriculum that grows the range from 3 m by success
    rate. 75% of goals include a heading.
  * Success = within 0.2 m, within 10° when a heading is asked for, and below 0.1 m/s; this ends the episode.
* **Actor observations** (8, deployable): goal in body frame ×0.2 (direction-preserving clip at 2), distance,
  sin/cos heading error (zeroed if no heading), has-heading flag, last action.
* **Critic observations** (15): the actor's plus true velocity, joint states and noise-free errors.
* **Actions** (2, in [−1, 1]): ×240 deg/s wheel speed, ×55° steering.
* **Rewards:** potential-based progress (distance + heading error weighted near the goal), +20 success,
  −10 failure, plus small time / action / action-rate costs.
* **Domain randomization:** camera noise and a random one-step action delay.
* **The observation layout lives in three places and must stay identical:**
  `ForkliftNavEnv._get_observations`, `Ros2 scripts/rl_policy.py::build_actor_obs`, and `OBS_LAYOUT` in
  `scripts/export_policy.py`. Change all three together, then retrain and re-export.
* The actor is `[128, 64]` ELU with `obs_normalization=False`, so the exported weights are the whole policy.
  Keep normalization off, or extend the exporter and `NumpyPolicy`.
* `forklift_b.usd` authors stiffness 100 on the continuous drive joint, a spring that fights the velocity
  drive. `robot.py` overrides it to 0. Isaac Sim still has it unless the user sets it in `Forklift_ROS2.usd`.
* The texture "asset can not be found" errors (`./Props/Source/Textures/...`) are cosmetic: the asset pack's
  textures were never copied. Ignore them.

### Commands (from `isaac_lab_forklift/`)

```bash
./scripts/train.sh                         # background, 1024 envs, 1000 it (~30 min); log in logs/
FOREGROUND=1 ./scripts/train.sh --num_envs 16 --max_iterations 3      # smoke test
./scripts/play.sh                          # Kit viewer, 16 envs; green bar = goal fork direction
~/IsaacLab/.venv/bin/python scripts/export_policy.py   # latest run -> ../Ros2 scripts/policies/forklift_nav.npz
source ~/IsaacLab/.venv/bin/activate
~/IsaacLab/isaaclab.sh -p scripts/drive_check.py       # sign / speed check with fixed commands
~/IsaacLab/isaaclab.sh -p scripts/eval_deploy.py --level 1   # closed loop through rl_policy.py + emulated tracker
```

`export_policy.py` needs no simulator. It reads `model_*.pt` (`actor_state_dict`, `mlp.{0,2,4}`) and
the run's `params/env.yaml`, and stores scales and tolerances as JSON metadata in the `.npz`. The ROS node
reads them from there, so don't hard-code them in the node.

### Current policy

Run `2026-10-18_11-46-17/model_999.pt`. `eval_deploy.py --level 1` (int pixels, heading low-pass,
latency): 100% of any-heading goals reached (9.5 s average), 92.7% of heading goals reached
(16.2 s), 0% failures, the rest timeouts at 45 s.

## ROS deployment

```bash
cd "Ros2 scripts" && ./run_rl_forklift.sh 1 2 3        # or: python3 rl_forklift_controller.py 1 --ros-args -p ...
ros2 topic pub --once /forklift_1/lift_target_pos geometry_msgs/msg/Pose2D "{x: 900.0, y: 360.0, theta: 30.0}"
ros2 topic echo /forklift_1/goal_status                # "reached: ..." / "failed: ..."
```

* **Parameters:** `policy_path`, `px_per_m` (65), `settle_ticks` (3), `rl_position_tolerance_px`
  (~16), `rl_heading_tolerance_deg` (15), `tag_heading_offset_deg` (0), `auto_heading_check` (true).
  `goal_timeout_sec` etc. are inherited.
* **`auto_heading_check`:** while driving nearly straight, the node compares the tag's actual motion
  with its believed heading. After two agreeing measurements it corrects the offset in 90° steps and logs
  the value to make permanent.
* **Inherited from `ForkliftController`:** mast control (`mast_target_pos`), `/stop`, the safe-area check,
  vision timeout and startup zero commands.
* **Testing without Isaac Sim:** a kinematic stand-in works well. Subscribe to `/forklift_1/ackermann`,
  integrate the rear-steer bicycle model (u = v·cos δ, ω = −v·sin δ / 1.39, v = speed·π/180·0.16, tag
  1.19 m ahead of the roller axle), and publish `vision_data` at 30 Hz as int pixels with `theta` = yaw + 180°
  through the 0.3 low-pass. Such a test reached all goals, including recovery from a wrong 180° offset.
  The script is not in the repo yet.

## Status and open items

* **Not yet verified:** the RL node in the real Isaac Sim warehouse with `vision_tracker.py`.
* Set the drive-joint stiffness to 0 in `Forklift_ROS2.usd` for sim-to-sim match (user action).
* Consider fixing `heading_offset_deg` in `forklift_controller.py` (user's code; ask first).
* Future: EOITAG task allocation in `fleet_controller.py`, using time-to-goal from `eval_deploy.py` (or the
  critic value) as travel cost; more robot types (e.g. `forklift_c`), each with a config in `robot.py` and
  its own registered task and policy; lift / pallet-pickup skills.

## Working rules

* Don't change the topic interface (`vision_data`, `lift_target_pos`, `ackermann`, `goal_status`, `stop`,
  `mast_target_pos`); the fleet / allocation layer depends on it.
* Retrain after changing env dynamics, observations or rewards. Re-export after every retrain. Check with
  `eval_deploy.py` before telling the user a policy works.
* Run training in the background (`train.sh` does) and watch it with a log filter; a full run takes ~30 min.
* `Issasim Models/*.usd` are the user's scene files. Read them with `pxr` (`Usd.Stage.Open`,
  `Sdf.Layer.ExportToString`); don't edit them without asking.
