#!/usr/bin/env python3

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Add ROS Python packages path if the shell has not sourced ROS setup.
ros_python_path = '/opt/ros/jazzy/lib/python3.12/site-packages'
if os.path.isdir(ros_python_path) and ros_python_path not in sys.path:
    sys.path.append(ros_python_path)

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose2D
from std_msgs.msg import Empty, String

import math
import threading

from rl_policy import pallet_entry_heading, staging_tag_pixels

FORKLIFT_IDS = (1, 2, 3)
PALLET_IDS = (1, 2, 3)
# Tracker theta -> fork direction. The forklift tag's top edge points along the forks (measured in
# Environment.usd), matching rl_forklift_controller.py's tag_heading_offset_deg default.
FORK_HEADING_OFFSET_DEG = 0.0
# Lift applied after a successful insert when `pallet` gets no lift argument.
DEFAULT_PICK_LIFT_M = 0.3

HELP_TEXT = """
Commands (positions are camera pixels, heading in degrees, lift in meters):
  Heading is the direction the forks point: 0 = right, 90 = up, 180 = left, -90 = down.
  <forklift> <x> <y> [heading_deg|-] [lift_m] Drive forklift to (x, y); '-' or omitted heading = any
  lift <forklift> <height_m>                  Set only the lift height
  pallet <pallet> <forklift> [lift_m]         Pick up: stage 1.5 m in front of the pallet's 1200 mm face,
                                              insert the forks, then lift to lift_m (default 0.3 m)
  place <forklift>                            Put the carried pallet down and back out of it
  stop <forklift|all>                         Stop driving immediately
  status                                      Show latest forklift and pallet poses
  help                                        Show this help
  quit                                        Exit the fleet controller

Examples:
  1 640 360            -> Forklift 1 drives to pixel (640, 360)
  2 900 200 90 1.2     -> Forklift 2 drives to (900, 200), forks facing up, lift to 1.2 m
  3 270 630 - 0.5      -> Forklift 3 drives to (270, 630), any heading, lift to 0.5 m
  lift 3 0.5           -> Forklift 3 lift to 0.5 m
  pallet 1 2 0.3       -> Forklift 2 picks up pallet 1 and lifts it 0.3 m
  2 640 360 90         -> (while carrying) drive the pallet to (640, 360), forks up
  place 2              -> lower the pallet and back out
"""


class FleetController(Node):
    def __init__(self):
        super().__init__('fleet_controller')

        self.forklift_poses = {}
        self.pallet_poses = {}

        # ------Subscribers------
        self.sub_forklifts = {}
        for fid in FORKLIFT_IDS:
            self.sub_forklifts[fid] = self.create_subscription(
                Pose2D,
                f'/forklift_{fid}/vision_data',
                lambda msg, i=fid: self.forklift_callback(i, msg),
                10
            )

        # Pallet poses are only stored; they are sent to a forklift on request.
        self.sub_pallets = {}
        for pid in PALLET_IDS:
            self.sub_pallets[pid] = self.create_subscription(
                Pose2D,
                f'/pallet_{pid}/vision_data',
                lambda msg, i=pid: self.pallet_callback(i, msg),
                10
            )

        # ------Publishers------
        self.forklift_target_pubs = {}
        self.forklift_lift_target_pubs = {}
        self.forklift_mast_target_pubs = {}
        self.forklift_stop_pubs = {}
        self.forklift_place_pubs = {}
        # Lift height to command once a forklift reports "reached: inserted".
        self.pending_pick_lift = {}
        for fid in FORKLIFT_IDS:
            self.forklift_target_pubs[fid] = self.create_publisher(
                Pose2D, f'/forklift_{fid}/target_pos', 10
            )
            self.forklift_lift_target_pubs[fid] = self.create_publisher(
                Pose2D, f'/forklift_{fid}/lift_target_pos', 10
            )
            self.forklift_mast_target_pubs[fid] = self.create_publisher(
                Twist, f'/forklift_{fid}/mast_target_pos', 10
            )
            self.forklift_stop_pubs[fid] = self.create_publisher(
                Empty, f'/forklift_{fid}/stop', 10
            )
            self.forklift_place_pubs[fid] = self.create_publisher(
                Empty, f'/forklift_{fid}/place', 10
            )

        self.sub_goal_status = {}
        for fid in FORKLIFT_IDS:
            self.sub_goal_status[fid] = self.create_subscription(
                String,
                f'/forklift_{fid}/goal_status',
                lambda msg, i=fid: self.goal_status_callback(i, msg),
                10
            )

        self.get_logger().info("============================================================")
        self.get_logger().info(" Fleet Controller Node Started (manual command mode)")
        self.get_logger().info(" Forklifts stay idle until you enter a command below.")
        self.get_logger().info("============================================================")

    def forklift_callback(self, forklift_id: int, msg: Pose2D):
        self.forklift_poses[forklift_id] = msg

    def pallet_callback(self, pallet_id: int, msg: Pose2D):
        self.pallet_poses[pallet_id] = msg

    def goal_status_callback(self, forklift_id: int, msg: String):
        self.get_logger().info(f"[Forklift {forklift_id}] Goal {msg.data}")
        height = self.pending_pick_lift.pop(forklift_id, None)
        if height is None:
            return
        if msg.data.startswith('reached: inserted'):
            self.send_lift_target(forklift_id, height)
        elif msg.data.startswith('reached: staging'):
            self.pending_pick_lift[forklift_id] = height  # still waiting for the insert
        else:
            self.get_logger().warn(f"[Forklift {forklift_id}] Pick-up did not finish; lift not raised.")

    # ------Command senders------
    def send_drive_target(self, forklift_id: int, x: float, y: float, heading_deg: float = math.nan):
        target = Pose2D(x=float(x), y=float(y), theta=float(heading_deg))
        self.forklift_target_pubs[forklift_id].publish(target)
        self.forklift_lift_target_pubs[forklift_id].publish(target)
        heading_txt = 'any' if math.isnan(heading_deg) else f"{heading_deg:.1f}°"
        self.get_logger().info(
            f"[Forklift {forklift_id}] Drive target: x={x:.1f}, y={y:.1f}, heading={heading_txt}"
        )

    def send_lift_target(self, forklift_id: int, height: float):
        cmd = Twist()
        cmd.linear.z = float(height)
        self.forklift_mast_target_pubs[forklift_id].publish(cmd)
        self.get_logger().info(f"[Forklift {forklift_id}] Lift target: {height:.2f} m")

    def send_place(self, forklift_id: int):
        self.pending_pick_lift.pop(forklift_id, None)
        self.forklift_place_pubs[forklift_id].publish(Empty())
        self.get_logger().info(f"[Forklift {forklift_id}] Place requested")

    def send_stop(self, forklift_id: int):
        self.forklift_stop_pubs[forklift_id].publish(Empty())
        self.get_logger().info(f"[Forklift {forklift_id}] Stop requested")

    # ------Command parsing------
    def handle_command(self, line: str) -> bool:
        """Parse and execute one user command. Returns False when the user wants to quit."""
        parts = line.strip().split()
        if not parts:
            return True

        cmd = parts[0].lower()
        try:
            if cmd in ('quit', 'exit', 'q'):
                return False
            elif cmd in ('help', 'h', '?'):
                print(HELP_TEXT)
            elif cmd == 'status':
                self.print_status()
            elif cmd == 'stop':
                if len(parts) != 2:
                    raise ValueError("usage: stop <forklift|all>")
                targets = FORKLIFT_IDS if parts[1].lower() == 'all' else (self.parse_forklift(parts[1]),)
                for fid in targets:
                    self.pending_pick_lift.pop(fid, None)
                    self.send_stop(fid)
            elif cmd == 'place':
                if len(parts) != 2:
                    raise ValueError("usage: place <forklift>")
                self.send_place(self.parse_forklift(parts[1]))
            elif cmd == 'lift':
                if len(parts) != 3:
                    raise ValueError("usage: lift <forklift> <height_m>")
                self.send_lift_target(self.parse_forklift(parts[1]), float(parts[2]))
            elif cmd == 'pallet':
                if len(parts) not in (3, 4):
                    raise ValueError("usage: pallet <pallet> <forklift> [lift_m]")
                pallet_id = int(parts[1])
                fid = self.parse_forklift(parts[2])
                pose = self.pallet_poses.get(pallet_id)
                if pose is None:
                    raise ValueError(f"pallet {pallet_id} has not been seen by the camera yet")
                fl = self.forklift_poses.get(fid)
                if fl is None:
                    raise ValueError(f"forklift {fid} has not been seen by the camera yet")
                # Forks enter the 1200 mm face nearest the forklift; tips stage 1.5 m from that face.
                heading = pallet_entry_heading(pose.theta, (fl.x, fl.y), (pose.x, pose.y))
                sx, sy = staging_tag_pixels((pose.x, pose.y), heading, px_per_m=65.0)
                # The lift goes up only after the insert succeeds (raised forks cannot enter the pockets).
                self.pending_pick_lift[fid] = float(parts[3]) if len(parts) == 4 else DEFAULT_PICK_LIFT_M
                self.send_drive_target(fid, sx, sy, math.degrees(heading))
            else:
                if len(parts) not in (3, 4, 5):
                    raise ValueError("usage: <forklift> <x> <y> [heading_deg] [lift_m]  (type 'help')")
                fid = self.parse_forklift(parts[0])
                x, y = float(parts[1]), float(parts[2])
                heading_deg = math.nan
                if len(parts) >= 4 and parts[3] != '-':
                    heading_deg = float(parts[3])
                self.send_drive_target(fid, x, y, heading_deg)
                if len(parts) == 5:
                    self.send_lift_target(fid, float(parts[4]))
        except ValueError as e:
            print(f"  ! {e}")
        return True

    @staticmethod
    def parse_forklift(token: str) -> int:
        fid = int(token)
        if fid not in FORKLIFT_IDS:
            raise ValueError(f"forklift id must be one of {FORKLIFT_IDS}")
        return fid

    def print_status(self):
        print("  Forklifts (heading = direction the forks point):")
        for fid in FORKLIFT_IDS:
            p = self.forklift_poses.get(fid)
            if p is None:
                print(f"    {fid}: not seen")
                continue
            forks = math.degrees(math.atan2(math.sin(p.theta + math.radians(FORK_HEADING_OFFSET_DEG)),
                                            math.cos(p.theta + math.radians(FORK_HEADING_OFFSET_DEG))))
            print(f"    {fid}: x={p.x:.1f}, y={p.y:.1f}, heading={forks:.1f}°")
        print("  Pallets:")
        for pid in PALLET_IDS:
            p = self.pallet_poses.get(pid)
            print(f"    {pid}: " + (f"x={p.x:.1f}, y={p.y:.1f}, heading={math.degrees(p.theta):.1f}°" if p else "not seen"))


def main(args=None):
    rclpy.init(args=args)
    fleet_controller = None
    try:
        fleet_controller = FleetController()
        spin_thread = threading.Thread(target=rclpy.spin, args=(fleet_controller,), daemon=True)
        spin_thread.start()

        if not sys.stdin.isatty():
            fleet_controller.get_logger().warn(
                "No interactive terminal attached; send goals with 'ros2 topic pub' instead."
            )
            spin_thread.join()
            return

        print(HELP_TEXT)
        while rclpy.ok():
            try:
                line = input("fleet> ")
            except EOFError:
                break
            if not fleet_controller.handle_command(line):
                break
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        if fleet_controller is not None:
            fleet_controller.get_logger().info("Shutting down Fleet Controller...")
            fleet_controller.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
