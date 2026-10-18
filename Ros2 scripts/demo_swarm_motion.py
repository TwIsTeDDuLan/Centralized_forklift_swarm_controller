#!/usr/bin/env python3
# ==============================================================================
# Centralized Forklift Swarm - Multi-Forklift Capability Demonstration Script
# ==============================================================================
# Demonstrates full swarm actuation capabilities for Forklifts 1, 2, and 3:
#   • Ackermann Drive: Move Forward, Backward (Reverse), Steer Left, Steer Right
#   • Fork Mast Control: Smoothly Lift Up, Lower Down
#   • Synchronized Swarm Choreography (all 3 forklifts in unison)
#
# Usage:
#   python3 demo_swarm_motion.py               # Interactive CLI Menu
#   python3 demo_swarm_motion.py --auto        # Run full automated choreography
#   python3 demo_swarm_motion.py --forward     # Move all forklifts forward
#   python3 demo_swarm_motion.py --backward    # Move all forklifts backward
#   python3 demo_swarm_motion.py --lift-up     # Raise forks on all forklifts
#   python3 demo_swarm_motion.py --lift-down   # Lower forks on all forklifts
#   python3 demo_swarm_motion.py --stop        # Stop all forklifts
# ==============================================================================

import os
import sys
import time
import argparse

# Add ROS Python packages path if needed
ros_python_path = '/opt/ros/jazzy/lib/python3.12/site-packages'
if os.path.isdir(ros_python_path) and ros_python_path not in sys.path:
    sys.path.append(ros_python_path)

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from ackermann_msgs.msg import AckermannDriveStamped

C_RESET = '\033[0m'
C_BOLD = '\033[1m'
C_RED = '\033[0;31m'
C_GREEN = '\033[0;32m'
C_YELLOW = '\033[0;33m'
C_BLUE = '\033[0;34m'
C_CYAN = '\033[0;36m'
C_MAGENTA = '\033[0;35m'
C_GRAY = '\033[0;90m'

class SwarmDemoNode(Node):
    def __init__(self):
        super().__init__('swarm_demo_controller')

        self.forklift_ids = [1, 2, 3]

        # Publishers for each forklift
        self.pub_drive = {}
        self.pub_lift_raw = {}
        self.pub_mast_target = {}

        for fid in self.forklift_ids:
            # Ackermann steering drive publisher
            self.pub_drive[fid] = self.create_publisher(
                AckermannDriveStamped, f'/forklift_{fid}/ackermann', 10
            )
            # Direct lift velocity/position
            self.pub_lift_raw[fid] = self.create_publisher(
                Twist, f'/forklift_{fid}/lift', 10
            )
            # High-level mast target position (handled by Fork_Lift PID loop)
            self.pub_mast_target[fid] = self.create_publisher(
                Twist, f'/forklift_{fid}/mast_target_pos', 10
            )

        self.get_logger().info("Swarm Motion Demo node initialized for Forklifts 1, 2, and 3.")

    def send_drive(self, forklift_ids, speed: float, steering_deg: float):
        """Publish an AckermannDriveStamped command to specified forklifts."""
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steering_deg)

        for fid in forklift_ids:
            if fid in self.pub_drive:
                self.pub_drive[fid].publish(msg)

    def send_lift(self, forklift_ids, height_or_speed: float):
        """Publish lift commands (both direct velocity and target height) to specified forklifts."""
        msg = Twist()
        msg.linear.z = float(height_or_speed)

        for fid in forklift_ids:
            if fid in self.pub_mast_target:
                self.pub_mast_target[fid].publish(msg)
            if fid in self.pub_lift_raw:
                self.pub_lift_raw[fid].publish(msg)

    def stop_all(self, forklift_ids=None):
        """Bring all drive motors to an immediate safe halt."""
        if forklift_ids is None:
            forklift_ids = self.forklift_ids
        self.send_drive(forklift_ids, speed=0.0, steering_deg=0.0)

    def execute_motion_stream(self, forklift_ids, speed: float, steering_deg: float, duration_sec: float, rate_hz: float = 10.0):
        """Stream drive commands at rate_hz for continuous physics simulation."""
        interval = 1.0 / rate_hz
        end_time = time.time() + duration_sec
        while time.time() < end_time and rclpy.ok():
            self.send_drive(forklift_ids, speed, steering_deg)
            rclpy.spin_once(self, timeout_sec=interval)
            time.sleep(interval)
        self.stop_all(forklift_ids)

    def execute_lift_stream(self, forklift_ids, height: float, duration_sec: float, rate_hz: float = 10.0):
        """Stream lift commands for duration_sec."""
        interval = 1.0 / rate_hz
        end_time = time.time() + duration_sec
        while time.time() < end_time and rclpy.ok():
            self.send_lift(forklift_ids, height)
            rclpy.spin_once(self, timeout_sec=interval)
            time.sleep(interval)

    # --- High-Level Choreographed Demos ---

    def demo_forward(self, forklift_ids, duration=3.0, speed=40.0):
        print(f"{C_GREEN}▶ Moving Forklifts {forklift_ids} FORWARD (Speed: {speed}, Steering: 0° for {duration}s)...{C_RESET}")
        self.execute_motion_stream(forklift_ids, speed=speed, steering_deg=0.0, duration_sec=duration)
        print(f"{C_OK_TAG()} Forward drive completed.")

    def demo_backward(self, forklift_ids, duration=3.0, speed=-40.0):
        print(f"{C_YELLOW}◀ Moving Forklifts {forklift_ids} BACKWARD / REVERSE (Speed: {speed}, Steering: 0° for {duration}s)...{C_RESET}")
        self.execute_motion_stream(forklift_ids, speed=speed, steering_deg=0.0, duration_sec=duration)
        print(f"{C_OK_TAG()} Reverse drive completed.")

    def demo_steer_left(self, forklift_ids, duration=2.5, speed=30.0, angle=35.0):
        print(f"{C_BLUE}↰ Steering Forklifts {forklift_ids} LEFT (Speed: {speed}, Steering: +{angle}° for {duration}s)...{C_RESET}")
        self.execute_motion_stream(forklift_ids, speed=speed, steering_deg=angle, duration_sec=duration)
        print(f"{C_OK_TAG()} Left turn completed.")

    def demo_steer_right(self, forklift_ids, duration=2.5, speed=30.0, angle=-35.0):
        print(f"{C_BLUE}↱ Steering Forklifts {forklift_ids} RIGHT (Speed: {speed}, Steering: {angle}° for {duration}s)...{C_RESET}")
        self.execute_motion_stream(forklift_ids, speed=speed, steering_deg=angle, duration_sec=duration)
        print(f"{C_OK_TAG()} Right turn completed.")

    def demo_lift_up(self, forklift_ids, height=1.6, duration=3.0):
        print(f"{C_MAGENTA}▲ Raising Forks on Forklifts {forklift_ids} UP (Target Height: {height}m for {duration}s)...{C_RESET}")
        self.execute_lift_stream(forklift_ids, height=height, duration_sec=duration)
        print(f"{C_OK_TAG()} Lift Up completed.")

    def demo_lift_down(self, forklift_ids, height=0.0, duration=3.0):
        print(f"{C_CYAN}▼ Lowering Forks on Forklifts {forklift_ids} DOWN (Target Height: {height}m for {duration}s)...{C_RESET}")
        self.execute_lift_stream(forklift_ids, height=height, duration_sec=duration)
        print(f"{C_OK_TAG()} Lower Down completed.")

    def run_full_choreography(self):
        """Run synchronized autonomous choreography for all 3 forklifts."""
        fids = self.forklift_ids
        print(f"\n{C_BOLD}{C_CYAN}======================================================================{C_RESET}")
        print(f"{C_BOLD}{C_CYAN}   STARTING FULL SYNCHRONIZED SWARM DEMONSTRATION (FORKLIFTS 1, 2, 3) {C_RESET}")
        print(f"{C_BOLD}{C_CYAN}======================================================================{C_RESET}\n")

        time.sleep(1.0)
        # 1. Raise Forks
        print(f"{C_BOLD}[Stage 1/6] Synchronized Fork Lift Up{C_RESET}")
        self.demo_lift_up(fids, height=1.5, duration=3.0)
        time.sleep(0.5)

        # 2. Drive Forward
        print(f"\n{C_BOLD}[Stage 2/6] Synchronized Ackermann Forward Drive{C_RESET}")
        self.demo_forward(fids, duration=3.0, speed=40.0)
        time.sleep(0.5)

        # 3. S-Curve Maneuver (Steer Left, then Right)
        print(f"\n{C_BOLD}[Stage 3/6] Synchronized Steering Maneuver (Left Turn){C_RESET}")
        self.demo_steer_left(fids, duration=2.0, speed=30.0, angle=30.0)
        time.sleep(0.5)

        print(f"\n{C_BOLD}[Stage 4/6] Synchronized Steering Maneuver (Right Turn Counter-Steer){C_RESET}")
        self.demo_steer_right(fids, duration=2.0, speed=30.0, angle=-30.0)
        time.sleep(0.5)

        # 4. Reverse Driving
        print(f"\n{C_BOLD}[Stage 5/6] Synchronized Ackermann Reverse Drive Back{C_RESET}")
        self.demo_backward(fids, duration=3.0, speed=-40.0)
        time.sleep(0.5)

        # 5. Lower Forks
        print(f"\n{C_BOLD}[Stage 6/6] Synchronized Fork Lower Down{C_RESET}")
        self.demo_lift_down(fids, height=0.0, duration=3.0)

        self.stop_all(fids)
        print(f"\n{C_BOLD}{C_GREEN}======================================================================{C_RESET}")
        print(f"{C_BOLD}{C_GREEN}        SWARM CAPABILITY DEMONSTRATION COMPLETED SUCCESSFULLY!        {C_RESET}")
        print(f"{C_BOLD}{C_GREEN}======================================================================{C_RESET}\n")

def C_OK_TAG():
    return f"{C_GREEN}[OK]{C_RESET}"

def run_interactive_menu(node: SwarmDemoNode):
    target_fids = [1, 2, 3]

    while rclpy.ok():
        print(f"\n{C_BOLD}{C_CYAN}======================================================================{C_RESET}")
        print(f"{C_BOLD}{C_CYAN}         Forklift Swarm Capability Demonstration Controller           {C_RESET}")
        print(f"{C_BOLD}{C_CYAN}======================================================================{C_RESET}")
        print(f" Active Target Robots: {C_BOLD}{C_YELLOW}{target_fids}{C_RESET}")
        print(f"----------------------------------------------------------------------")
        print(f"  {C_BOLD}1{C_RESET})  Drive Forward (Ackermann)")
        print(f"  {C_BOLD}2{C_RESET})  Drive Backward / Reverse (Ackermann)")
        print(f"  {C_BOLD}3{C_RESET})  Steer Left (Forward + Left Turn)")
        print(f"  {C_BOLD}4{C_RESET})  Steer Right (Forward + Right Turn)")
        print(f"  {C_BOLD}5{C_RESET})  Raise Forks Smoothly (Mast Lift Up: 1.5m)")
        print(f"  {C_BOLD}6{C_RESET})  Lower Forks Smoothly (Mast Lift Down: 0.0m)")
        print(f"  {C_BOLD}7{C_RESET})  {C_RED}Emergency Stop All Motors{C_RESET}")
        print(f"  {C_BOLD}8{C_RESET})  {C_GREEN}Run Full Synchronized Choreography (All 6 Stages){C_RESET}")
        print(f"  {C_BOLD}9{C_RESET})  Change Target Forklifts (Currently: {target_fids})")
        print(f"  {C_BOLD}0{C_RESET})  Exit")
        print(f"======================================================================")

        try:
            choice = input(f"{C_BOLD}Enter command [0-9]: {C_RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting demo.")
            node.stop_all()
            break

        if choice == '1':
            node.demo_forward(target_fids, duration=2.5, speed=40.0)
        elif choice == '2':
            node.demo_backward(target_fids, duration=2.5, speed=-40.0)
        elif choice == '3':
            node.demo_steer_left(target_fids, duration=2.5, speed=30.0, angle=35.0)
        elif choice == '4':
            node.demo_steer_right(target_fids, duration=2.5, speed=30.0, angle=-35.0)
        elif choice == '5':
            node.demo_lift_up(target_fids, height=1.6, duration=3.0)
        elif choice == '6':
            node.demo_lift_down(target_fids, height=0.0, duration=3.0)
        elif choice == '7':
            print(f"{C_RED}[STOP] Halting all motors...{C_RESET}")
            node.stop_all(target_fids)
            print(f"{C_OK_TAG()} All motors stopped.")
        elif choice == '8':
            node.run_full_choreography()
        elif choice == '9':
            print("\nSelect target forklifts:")
            print("  a) All 3 Forklifts ([1, 2, 3])")
            print("  1) Forklift 1 only")
            print("  2) Forklift 2 only")
            print("  3) Forklift 3 only")
            sub_c = input("Choice [a/1/2/3]: ").strip().lower()
            if sub_c == '1':
                target_fids = [1]
            elif sub_c == '2':
                target_fids = [2]
            elif sub_c == '3':
                target_fids = [3]
            else:
                target_fids = [1, 2, 3]
            print(f"{C_OK_TAG()} Target updated to: {target_fids}")
        elif choice in ('0', 'q', 'exit'):
            print("Shutting down swarm demo controller.")
            node.stop_all()
            break
        else:
            print(f"{C_YELLOW}Invalid option. Please choose 0 to 9.{C_RESET}")

def main():
    parser = argparse.ArgumentParser(description="Multi-Forklift Swarm Capability Demonstration")
    parser.add_argument('--auto', action='store_true', help="Run full synchronized choreography sequence and exit")
    parser.add_argument('--forward', action='store_true', help="Move selected forklifts forward")
    parser.add_argument('--backward', action='store_true', help="Move selected forklifts backward")
    parser.add_argument('--steer-left', action='store_true', help="Steer selected forklifts left")
    parser.add_argument('--steer-right', action='store_true', help="Steer selected forklifts right")
    parser.add_argument('--lift-up', action='store_true', help="Raise forks up to 1.5m")
    parser.add_argument('--lift-down', action='store_true', help="Lower forks down to 0m")
    parser.add_argument('--stop', action='store_true', help="Stop all forklifts")
    parser.add_argument('--forklift', type=int, choices=[1, 2, 3], default=None, help="Target specific forklift ID (default: all 3)")
    parser.add_argument('--duration', type=float, default=2.5, help="Motion duration in seconds")
    parser.add_argument('--speed', type=float, default=40.0, help="Drive speed")

    args, _ = parser.parse_known_args()

    rclpy.init()
    demo_node = SwarmDemoNode()

    target_fids = [args.forklift] if args.forklift else [1, 2, 3]

    try:
        if args.auto:
            demo_node.run_full_choreography()
        elif args.forward:
            demo_node.demo_forward(target_fids, duration=args.duration, speed=args.speed)
        elif args.backward:
            demo_node.demo_backward(target_fids, duration=args.duration, speed=-args.speed)
        elif args.steer_left:
            demo_node.demo_steer_left(target_fids, duration=args.duration, speed=args.speed)
        elif args.steer_right:
            demo_node.demo_steer_right(target_fids, duration=args.duration, speed=args.speed)
        elif args.lift_up:
            demo_node.demo_lift_up(target_fids, height=1.6, duration=args.duration)
        elif args.lift_down:
            demo_node.demo_lift_down(target_fids, height=0.0, duration=args.duration)
        elif args.stop:
            demo_node.stop_all(target_fids)
            print(f"{C_RED}[STOP] Motors stopped.{C_RESET}")
        else:
            run_interactive_menu(demo_node)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupt received. Stopping all forklifts...")
        demo_node.stop_all()
    finally:
        demo_node.stop_all()
        demo_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
