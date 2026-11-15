"""
Test corridor-guided approach vs direct approach on fake_world.py.
Measures true tip positions in pallet frame at insertion.
"""
import math
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import rclpy
from geometry_msgs.msg import Pose2D
from std_msgs.msg import String

import fleet_controller as fc
from rl_policy import pallet_entry_heading, staging_tag_pixels

PX_PER_M = 65.0

def approach_tag_pixels(pallet_xy, fork_heading_rad, px_per_m=65.0, tip_standoff=3.5, tag_to_tip=2.518, face_offset=0.4):
    dist_px = (face_offset + tip_standoff + tag_to_tip) * px_per_m
    ux, uy = math.cos(fork_heading_rad), -math.sin(fork_heading_rad)
    return pallet_xy[0] - dist_px * ux, pallet_xy[1] - dist_px * uy

def run_test(use_corridor=True):
    rclpy.init()
    fleet = fc.FleetController()
    statuses = []
    fleet.create_subscription(String, '/forklift_1/goal_status', lambda m: statuses.append((m.data, time.time())), 10)
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(fleet)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()

    def wait_for(prefix, timeout_sec=50.0):
        t0 = time.time()
        while time.time() - t0 < timeout_sec:
            for item in list(statuses):
                text, ts = item
                if ts >= t0:
                    if text.startswith(prefix) or text.startswith('failed') or text.startswith('rejected'):
                        statuses.remove(item)
                        print(f"  [{time.time() - t0:5.1f} s] status: {text}", flush=True)
                        return text
            time.sleep(0.05)
        print(f"  TIMEOUT waiting for {prefix}", flush=True)
        return None

    time.sleep(3.0)
    p_pal = fleet.pallet_poses.get(1)
    p_fl = fleet.forklift_poses.get(1)
    if not p_pal or not p_fl:
        print("Pallet or Forklift not seen!")
        fleet.destroy_node()
        rclpy.try_shutdown()
        return

    heading = pallet_entry_heading(p_pal.theta, (p_fl.x, p_fl.y), (p_pal.x, p_pal.y))
    sx, sy = staging_tag_pixels((p_pal.x, p_pal.y), heading, px_per_m=PX_PER_M)
    h_deg = math.degrees(heading)

    print(f"=== Running Pick Test (use_corridor={use_corridor}) ===")
    print(f"Pallet: ({p_pal.x:.1f}, {p_pal.y:.1f}), Forklift: ({p_fl.x:.1f}, {p_fl.y:.1f})")
    print(f"Staging pose: ({sx:.1f}, {sy:.1f}), heading: {h_deg:.1f}°")

    if use_corridor:
        ax, ay = approach_tag_pixels((p_pal.x, p_pal.y), heading, px_per_m=PX_PER_M, tip_standoff=3.5)
        print(f"Step 1a: Approach Waypoint: ({ax:.1f}, {ay:.1f}), heading: {h_deg:.1f}°")
        fleet.send_drive_target(1, ax, ay, h_deg)
        wait_for("reached:", timeout_sec=45.0)

    print(f"Step 1b: Staging Waypoint: ({sx:.1f}, {sy:.1f}), heading: {h_deg:.1f}°")
    fleet.pending_pick_lift[1] = 0.3
    fleet.send_drive_target(1, sx, sy, h_deg)
    wait_for("reached: staging", timeout_sec=40.0)
    st = wait_for("reached: inserted", timeout_sec=30.0)
    print(f"Insert finished: {st}")

    executor.shutdown()
    spinner.join(timeout=2)
    fleet.destroy_node()
    rclpy.try_shutdown()

if __name__ == '__main__':
    use_corr = '--corridor' in sys.argv
    run_test(use_corridor=use_corr)
