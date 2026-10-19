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
  rl_forklift_controller.py  two-policy RL node: nav to staging, then insert (same topics + /scan)
  rl_policy.py             numpy MLP + nav/insert observation builders (no ROS imports)
  policies/forklift_nav.npz, forklift_insert.npz
  attach_fork_lidar.py     3x64 fork lidar -> /forklift_N/scan; run in Isaac Sim's Script Editor (no USD edits)
  tests/fake_world.py      kinematic forklift 1 + pallet 1 + camera + mast + lidar (no Isaac Sim)
  run_fake_pick_test.sh    fleet pick -> deliver -> place against fake_world + the real RL node
  fleet_controller.py, run_*.sh, cmds.txt (example topic commands)
isaac_lab_forklift/      Isaac Lab package `forklift_lab`; see its README.md
  forklift_lab/robot.py    articulation cfg, fork-tip / lidar mount constants
  forklift_lab/pallet.py   EUR pocket geometry + o3dyn visual spawn
  forklift_lab/tasks/      Isaac-Forklift-Nav-Direct, Isaac-Forklift-Insert-Direct
  scripts/                 train.sh, play.sh, export_policy.py, eval_deploy.py, eval_insert.py,
                           check_fork_lidar.py (headless warehouse lidar check, run with ~/issacsim/python.sh)
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
* **Insert env** `forklift_lab/tasks/forklift_insert_env.py` (`ForkliftInsertEnv`):
  * Starts in a 1–2 m staging band (tips in front of the 1200 mm face). Entry is the long face;
    800 mm openings do not fit the 0.59 m tine spacing. Pockets: y ±0.275 m, 0.40 m wide, z 0.02–0.10 m.
  * Actor (196): 3×64 fork-mounted ranges / 4 m, last action (3), lift joint in [−1, 1].
  * Critic: actor + tip-to-pocket vectors, depth / lateral / yaw / height errors, contact force.
  * Actions (3): ×240 deg/s speed, ×55° steer, ×0.3 m/s lift (USD limits −0.15…2.0 m).
  * Rewards: potential on Euclidean tip-to-pocket distance, +20 full insert, −50 face hit (contact
    with a face stringer while tips are not in a pocket opening) and terminate, −10 other failure.
  * Success = both tips past 0.40 m into the pockets (reward target 0.70 m), inside the pocket AABB, no face contact.
  * Stacks: 1–3 decks (pitch 0.144 m), random target level. `o3dyn_pallet.usd` is visual only
    (authored scale is not EUR; collision_boxes wraps the load).
  * Insert observation layout lives in three places: `ForkliftInsertEnv._get_observations`,
    `rl_policy.py::build_insert_obs`, and `INSERT_OBS` in `export_policy.py`.

### Commands (from `isaac_lab_forklift/`)

```bash
./scripts/train.sh                         # nav, background, 1024 envs, 1000 it (~30 min); log in logs/
TASK=Isaac-Forklift-Insert-Direct ./scripts/train.sh --num_envs 256
FOREGROUND=1 TASK=Isaac-Forklift-Insert-Direct ./scripts/train.sh --num_envs 16 --max_iterations 3
TASK=Isaac-Forklift-Insert-Direct ./scripts/play.sh
~/IsaacLab/.venv/bin/python scripts/export_policy.py
~/IsaacLab/.venv/bin/python scripts/export_policy.py --experiment forklift_insert
~/IsaacLab/isaaclab.sh -p scripts/eval_deploy.py --level 1
~/IsaacLab/isaaclab.sh -p scripts/eval_insert.py --level 1
```

`export_policy.py` needs no simulator. It reads `model_*.pt` (`actor_state_dict`, `mlp.{0,2,4}`) and
the run's `params/env.yaml`, and stores scales and tolerances as JSON metadata in the `.npz`. The ROS node
reads them from there, so don't hard-code them in the node.

### Current policies

Nav: run `2026-10-18_11-46-17/model_999.pt`. `eval_deploy.py --level 1` (int pixels, heading low-pass,
latency): 100% of any-heading goals reached (9.5 s average), 92.7% of heading goals reached
(16.2 s), 0% failures, the rest timeouts at 45 s.

Insert: run `2026-10-19_00-31-36/model_1500.pt` (numpy `forklift_insert.npz`). `eval_insert.py`
(env 3×64 scan + numpy actor, no extra range noise):

| Curriculum | Reached | Face-hit | Failed | Timeout | Mean time |
|---|---|---|---|---|---|
| level 0 (staging band, single deck) | 100.0% | 0% | 0% | 0% | 3.3 s |
| level 1 (max pose/lift noise, 1–3 decks) | 77.9% | 0% | 0% | 22.1% | 3.3 s |

Level-1 timeouts sit ~0.45 m into the face with ~5 cm mean lateral error — a cocked one-tine-in
approach, not a collision. Do not claim warehouse insertion works until it is run with
`vision_tracker.py` and `attach_fork_lidar.py`.

## ROS deployment

Start Isaac Sim from a shell with `source /opt/ros/jazzy/setup.bash` (its ROS 2 bridge and the lidar need it).

```bash
cd "Ros2 scripts"
./run_tracker.sh                     # vision
./run_rl_forklift.sh 1 2 3           # RL nodes
# Isaac Sim Script Editor: open + Run attach_fork_lidar.py   (./run_fork_lidar.sh prints the steps)
./run_fleet_controller.sh            # fleet> pallet 1 2 0.3 | 2 900 500 90 | place 2 | 1 640 360 - | stop all
ros2 topic echo /forklift_1/goal_status
./run_fake_pick_test.sh [x y yaw]    # whole pick cycle without Isaac Sim (ROS_DOMAIN_ID 81)
```

* **Node states** (`rl_forklift_controller.py`): `nav_staging` → `insert` → idle and carrying →
  (`lift_target_pos` = delivery, no insert) → `/forklift_N/place` → `place` (lower, back out) → idle.
  * Statuses: `reached: staging`, `reached: inserted (...)`, `reached: delivered (...)`,
    `reached: placed (...)`, `failed: ...`, `rejected: ...`.
  * A nav goal counts as reached only after `settle_ticks` with the tag nearly still (`settle_motion_px` 4,
    `settle_turn_deg` 3). Without this, staging fired mid-turn, 30° off.
  * Staging tolerance = nav training (0.2 m × 1.25 ≈ 16 px, 15°). 0.5 m / 30° put the insert outside its
    training band and it went in crooked.
  * Insert travel = tip-to-face distance (nearest level-channel hit × cos(azimuth), when the insert starts)
    + `insert_depth_m` 0.65. No face in the scan → `failed`, never inserts blind.
  * On `inserted` the node holds the lift (clears the mast target); `mast_target_pos` then lifts.
    `place` lowers to the insert height − `place_lower_m` (0.05) and backs out `place_backout_m` (2.5 m)
    counterweight-first.
* **Fleet** `pallet P F [lift]`:
  * The pallet tag's top edge is the normal of a 1200 mm face (measured). The forks enter along ±theta,
    whichever face looks toward the forklift (`rl_policy.pallet_entry_heading`).
  * The staging tag pose puts the tips 1.5 m from that face (`staging_tag_pixels`, face 0.4 m from the
    pallet tag).
  * The lift (default 0.3 m) is sent only after `reached: inserted`.
  * `FORK_HEADING_OFFSET_DEG` = 0.
* **`attach_fork_lidar.py`:**
  * Hooks PhysX step events (never a blocking loop: that froze Isaac Sim) and reads the lift pose from
    PhysX (USD xforms are stale under Fabric).
  * Uses its own rclpy context. Running it again replaces the old instance.
  * Skips hits on the floor, on the own forklift and on `collision_boxes` (the pallet load box), because
    the training lidar only models the deck.
  * `check_fork_lidar.py` result: warehouse scan at staging = training model within 1 mm, at 10 Hz.
* **Parameters:** `policy_path`, `insert_policy_path`, `auto_insert`, `px_per_m` 65, `settle_*`,
  `rl_position_tolerance_px`, `rl_heading_tolerance_deg` 15, `tag_heading_offset_deg` 0,
  `auto_heading_check`, `insert_depth_m`, `insert_timeout_sec` 20, `place_*`, plus inherited ones.
* **`auto_heading_check`:** while driving nearly straight, compares the tag's motion with its believed
  heading and corrects the offset in 90° steps after two agreeing measurements.
* **Fake-world results** (5 start poses): every cycle stage → insert → lift → deliver → place succeeded;
  tips 0.43–0.69 m deep, within 0.06 m of the pocket centres. It is kinematic: no contact, no pallet physics.

## Status and open items

* **Not yet verified:** the RL node in the real Isaac Sim warehouse with `vision_tracker.py`.
* Set the drive-joint stiffness to 0 in `Forklift_ROS2.usd` for sim-to-sim match (user action).
* Consider fixing `heading_offset_deg` in `forklift_controller.py` (user's code; ask first).
* Future: EOITAG task allocation in `fleet_controller.py`, using time-to-goal from `eval_deploy.py` (or the
  critic value) as travel cost; more robot types (e.g. `forklift_c`), each with a config in `robot.py` and
  its own registered task and policy; lift after insert / place-down skills.
* Insert policy eval is in `eval_insert.py` (100% at level 0, 78% at level 1, 0% face hits).
* **Not yet verified in the warehouse:** insert / lift / carry / place with real contact. Watch for the
  pallet (a dynamic rigid body) being pushed, and for the load sliding while carrying.
* `/World/o3dyn_pallet_3` has no ArUco mesh, so the tracker never sees pallet 3.

## Working rules

* Don't change the topic interface (`vision_data`, `lift_target_pos`, `ackermann`, `goal_status`, `stop`,
  `mast_target_pos`); the fleet / allocation layer depends on it. `/forklift_N/scan` and `/forklift_N/place` are additive.
* Retrain after changing env dynamics, observations or rewards. Re-export after every retrain. Check with
  `eval_deploy.py` / `eval_insert.py` before telling the user a policy works.
* Run training in the background (`train.sh` does) and watch it with a log filter; a full run takes ~30 min.
* `Issasim Models/*.usd` are the user's scene files. Read them with `pxr` (`Usd.Stage.Open`,
  `Sdf.Layer.ExportToString`); don't edit them without asking.
