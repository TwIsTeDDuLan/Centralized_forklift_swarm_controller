"""Drive the fleet controller's pick / deliver / place commands against fake_world.py + the real RL node.

Run through ./run_fake_pick_test.sh (starts fake_world.py and rl_forklift_controller.py 1 on a private
ROS_DOMAIN_ID). Prints each goal_status; fake_world.log has the true fork-tip positions.
"""
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import rclpy
from std_msgs.msg import String

import fleet_controller as fc

rclpy.init()
fleet = fc.FleetController()
statuses = []
fleet.create_subscription(String, '/forklift_1/goal_status', lambda m: statuses.append(m.data), 10)
executor = rclpy.executors.SingleThreadedExecutor()
executor.add_node(fleet)
spinner = threading.Thread(target=executor.spin, daemon=True)
spinner.start()


def wait_for(prefix, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        for s in statuses:
            if s.startswith(prefix) or s.startswith('failed') or s.startswith('rejected'):
                statuses.remove(s)
                print(f"  [{time.time() - t0:5.1f} s] status: {s}", flush=True)
                return s
        time.sleep(0.1)
    print(f"  TIMEOUT waiting for {prefix}", flush=True)
    return None


time.sleep(4)
print("> status"); fleet.handle_command("status")
print("> pallet 1 1 0.3", flush=True); fleet.handle_command("pallet 1 1 0.3")
wait_for("reached: staging", 150)
if (s := wait_for("reached: inserted", 40)) and s.startswith("reached"):
    time.sleep(4)  # fleet raises the lift on "inserted"
    print("> 1 900 500 90  (deliver)", flush=True); fleet.handle_command("1 900 500 90")
    if (s := wait_for("reached: delivered", 150)) and s.startswith("reached"):
        print("> place 1", flush=True); fleet.handle_command("place 1")
        wait_for("reached: placed", 40)
executor.shutdown()
spinner.join(timeout=2)
fleet.destroy_node()
rclpy.try_shutdown()
