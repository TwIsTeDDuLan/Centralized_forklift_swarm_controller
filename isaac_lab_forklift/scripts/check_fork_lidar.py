"""Headless check of Ros2 scripts/attach_fork_lidar.py in the real warehouse scene (Environment.usd).

Checks that /forklift_1..3/scan publish while the simulation keeps running, and compares a virtual scan
1.5 m in front of pallet 1 with the insert policy's training lidar model, ray by ray.

    source /opt/ros/jazzy/setup.bash      # Isaac Sim's ROS 2 bridge needs it, as for the real scene
    ~/issacsim/python.sh scripts/check_fork_lidar.py
"""
import math
import os
import sys
import time

os.environ["ROS_DOMAIN_ID"] = "79"
from isaacsim import SimulationApp

app = SimulationApp({"headless": True})
from isaacsim.core.utils.extensions import enable_extension

enable_extension("isaacsim.ros2.bridge")
app.update()

import omni.timeline
import omni.usd
import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ctx = omni.usd.get_context()
ctx.open_stage(f"{PROJ}/Issasim Models/Environment.usd")
for _ in range(200):
    app.update()
    if not ctx.get_stage_loading_status()[2]:
        break
omni.timeline.get_timeline_interface().play()
for _ in range(30):
    app.update()

sys.path.insert(0, f"{PROJ}/Ros2 scripts")
import attach_fork_lidar as fl

lidar = fl.start()

import rclpy
from sensor_msgs.msg import LaserScan

c = rclpy.Context()
c.init()
node = rclpy.create_node("lidar_test", context=c)
got = {}
for fid in (1, 2, 3):
    node.create_subscription(LaserScan, f"/forklift_{fid}/scan", lambda m, i=fid: got.setdefault(i, []).append(m), 10)
from rclpy.executors import SingleThreadedExecutor

ex = SingleThreadedExecutor(context=c)
ex.add_node(node)
tl = omni.timeline.get_timeline_interface()
t0, w0 = tl.get_current_time(), time.time()
for _ in range(240):
    app.update()
    ex.spin_once(timeout_sec=0.0)
for _ in range(20):
    ex.spin_once(timeout_sec=0.05)
print(f"\n=== sim time advanced {tl.get_current_time() - t0:.2f} s in {time.time() - w0:.1f} s wall")
for fid in (1, 2, 3):
    msgs = got.get(fid, [])
    if not msgs:
        print(f"forklift {fid}: NO scans")
        continue
    r = np.array(msgs[-1].ranges).reshape(3, 64)
    print(f"forklift {fid}: {len(msgs)} scans, {len(msgs[-1].ranges)} ranges; per-channel min "
          f"{np.round(r.min(1), 2)} (4.0 = nothing)")

# ---- virtual lidar 1.5 m (tips) in front of pallet 1's -X entry face, forks along +X ----
face_x = 9.38
tips_x = face_x - 1.5
lidar_x = tips_x + 0.138
q = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))  # lift -Y -> world +X
for lift_z in (-0.146, -0.10):
    pos = (lidar_x - 2.0, 0.0, lift_z)
    lidar.lift_pose = lambda path, pos=pos: (pos, q)
    lidar.forklifts[99] = ("x", "/nothing/")
    real = np.array(lidar.scan(99)).reshape(3, 64)

    # training box model (pallet frame: face at x=0, +X in, y left), same rays
    oz = lift_z + 0.28
    boxes = [((0, 0.05), (y - 0.05, y + 0.05), (0, 0.144)) for y in (-0.55, 0.0, 0.55)]
    boxes += [((0, 0.8), (-0.6, 0.6), (0, 0.02)), ((0, 0.8), (-0.6, 0.6), (0.10, 0.144)), ((0.76, 0.8), (-0.6, 0.6), (0, 0.144))]
    o = np.array([-1.5 + 0.138, 0.0, oz])
    train = []
    for d_l in fl.ray_dirs_local():
        d = np.array(fl.quat_rotate(q, d_l))
        best = 4.0
        for bx in boxes:
            lo = np.array([b[0] for b in bx]); hi = np.array([b[1] for b in bx])
            with np.errstate(divide="ignore", invalid="ignore"):
                t1, t2 = (lo - o) / d, (hi - o) / d
            tn, tf = np.nanmax(np.minimum(t1, t2)), np.nanmin(np.maximum(t1, t2))
            if tf >= max(tn, 0) and tn < best:
                best = max(tn, 0)
        train.append(best)
    train = np.array(train).reshape(3, 64)
    diff = np.abs(real - train)
    print(f"\nvirtual staging scan, lift z {lift_z} (lidar z {oz:.2f}):")
    for ch, name in enumerate(("-8 deg", " 0 deg", "+8 deg")):
        print(f"  {name}: warehouse {np.round(real[ch, ::8], 2)}")
        print(f"          training  {np.round(train[ch, ::8], 2)}   mean |diff| {diff[ch].mean():.3f} m")

fl.stop()
node.destroy_node()
c.shutdown()
app.close()
