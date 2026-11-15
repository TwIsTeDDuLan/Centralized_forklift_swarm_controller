#!/usr/bin/env python3
"""
Interactive & automated test harness for the RL policies (Nav + Insert + Place)
running against the live warehouse stack (or fake_world).

Supports both numbered menu options and direct fleet CLI commands:
  - pallet <P> <F> [lift_m]         Pick pallet (stage -> insert -> lift)
  - <F> <x> <y> [heading_deg|-]     Drive forklift to pixel target
  - place <F>                       Place pallet & back out
  - lift <F> <height_m>             Command mast height
  - stop <F|all>                    Emergency stop
  - status                          Swarm status & preflight check
  - monitor                         Live streaming telemetry dashboard

CLI flags:
  ./test_rl_policies.sh                     # Interactive CLI prompt / menu
  ./test_rl_policies.sh --check             # Preflight check of topics & sensors
  ./test_rl_policies.sh --nav 1 640 360 90  # Test Go-To-Pose on forklift 1
  ./test_rl_policies.sh --pick 1 1 0.3      # Test pallet pick (stage -> insert -> lift)
  ./test_rl_policies.sh --cycle 1 1         # Full pick -> deliver -> place cycle
  ./test_rl_policies.sh --place 1           # Place down & back out
  ./test_rl_policies.sh --monitor           # Live monitor of all forklifts & pallets
  ./test_rl_policies.sh --stop [all|1|2|3]  # Stop forklift(s)
"""

import argparse
import math
import os
import sys
import threading
import time

# Ensure imports from Ros2 scripts/ work
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# Source ROS 2 if needed
ROS_PYTHON_PATH = "/opt/ros/jazzy/lib/python3.12/site-packages"
if os.path.isdir(ROS_PYTHON_PATH) and ROS_PYTHON_PATH not in sys.path:
    sys.path.append(ROS_PYTHON_PATH)

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose2D, Twist
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Empty, String

from rl_policy import pallet_entry_heading, staging_tag_pixels, pocket_corridor_errors

FORKLIFT_IDS = (1, 2, 3)
PALLET_IDS = (1, 2, 3)
PX_PER_M = 65.0
COLLISION_DISTANCE_PX = 80.0  # ~1.23 m; warning threshold between forklifts

# ANSI colors
C_RESET = '\033[0m'
C_BOLD = '\033[1m'
C_RED = '\033[0;31m'
C_GREEN = '\033[0;32m'
C_YELLOW = '\033[0;33m'
C_BLUE = '\033[0;34m'
C_CYAN = '\033[0;36m'
C_MAGENTA = '\033[0;35m'
C_GRAY = '\033[0;90m'


class RLTesterNode(Node):
    def __init__(self):
        super().__init__('rl_policy_tester')

        self.forklift_poses = {}
        self.forklift_ackermann = {}
        self.pallet_poses = {}
        self.last_status = {}
        self.status_history = {}
        self.lidar_active = {}

        # Publishers
        self.target_pubs = {}
        self.target_legacy_pubs = {}
        self.mast_pubs = {}
        self.place_pubs = {}
        self.stop_pubs = {}

        for fid in FORKLIFT_IDS:
            self.last_status[fid] = None
            self.status_history[fid] = []
            self.lidar_active[fid] = False
            self.forklift_ackermann[fid] = (0.0, 0.0, time.time())

            # Subscribers
            self.create_subscription(
                Pose2D, f'/forklift_{fid}/vision_data',
                lambda m, i=fid: self._on_forklift_pose(i, m), 10
            )
            self.create_subscription(
                AckermannDriveStamped, f'/forklift_{fid}/ackermann',
                lambda m, i=fid: self._on_ackermann(i, m), 10
            )
            self.create_subscription(
                String, f'/forklift_{fid}/goal_status',
                lambda m, i=fid: self._on_goal_status(i, m), 10
            )
            self.create_subscription(
                LaserScan, f'/forklift_{fid}/scan',
                lambda m, i=fid: self._on_scan(i, m), 10
            )

            # Publishers: both lift_target_pos and target_pos for compatibility
            self.target_pubs[fid] = self.create_publisher(Pose2D, f'/forklift_{fid}/lift_target_pos', 10)
            self.target_legacy_pubs[fid] = self.create_publisher(Pose2D, f'/forklift_{fid}/target_pos', 10)
            self.mast_pubs[fid] = self.create_publisher(Twist, f'/forklift_{fid}/mast_target_pos', 10)
            self.place_pubs[fid] = self.create_publisher(Empty, f'/forklift_{fid}/place', 10)
            self.stop_pubs[fid] = self.create_publisher(Empty, f'/forklift_{fid}/stop', 10)

        for pid in PALLET_IDS:
            self.create_subscription(
                Pose2D, f'/pallet_{pid}/vision_data',
                lambda m, i=pid: self._on_pallet_pose(i, m), 10
            )

    def _on_forklift_pose(self, fid: int, msg: Pose2D):
        self.forklift_poses[fid] = (msg.x, msg.y, msg.theta, time.time())

    def _on_ackermann(self, fid: int, msg: AckermannDriveStamped):
        self.forklift_ackermann[fid] = (msg.drive.speed, msg.drive.steering_angle, time.time())

    def _on_pallet_pose(self, pid: int, msg: Pose2D):
        self.pallet_poses[pid] = (msg.x, msg.y, msg.theta, time.time())

    def _on_goal_status(self, fid: int, msg: String):
        self.last_status[fid] = (msg.data, time.time())
        self.status_history[fid].append((msg.data, time.time()))

    def _on_scan(self, fid: int, msg: LaserScan):
        self.lidar_active[fid] = True

    def check_collision_risk(self, fid: int, target_x: float, target_y: float):
        """Warn user if target coordinate is very close to another vehicle."""
        for other_fid in FORKLIFT_IDS:
            if other_fid == fid:
                continue
            pose = self.forklift_poses.get(other_fid)
            if pose and (time.time() - pose[3] < 3.0):
                dist = math.hypot(target_x - pose[0], target_y - pose[1])
                if dist < COLLISION_DISTANCE_PX:
                    print(f"  {C_YELLOW}[COLLISION WARNING]{C_RESET} Target ({target_x:.0f}, {target_y:.0f}) is "
                          f"only {dist:.0f} px ({dist / PX_PER_M:.2f} m) from Forklift {other_fid} at "
                          f"({pose[0]:.0f}, {pose[1]:.0f})! They may collide.")

    def wait_for_subscribers(self, pub, timeout: float = 2.0) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if pub.get_subscription_count() > 0:
                return True
            time.sleep(0.05)
        return False

    def send_drive(self, fid: int, x: float, y: float, theta_deg: float = math.nan):
        """Publish drive target to both topics with multi-packet burst to guarantee delivery."""
        self.check_collision_risk(fid, x, y)
        msg = Pose2D()
        msg.x = float(x)
        msg.y = float(y)
        msg.theta = float(theta_deg)

        self.wait_for_subscribers(self.target_pubs[fid], timeout=1.5)

        # Send burst of 5 messages over ~200 ms
        for _ in range(5):
            self.target_pubs[fid].publish(msg)
            self.target_legacy_pubs[fid].publish(msg)
            time.sleep(0.04)

    def send_mast(self, fid: int, z_m: float):
        """Publish mast height target."""
        msg = Twist()
        msg.linear.z = float(z_m)
        self.wait_for_subscribers(self.mast_pubs[fid], timeout=1.5)
        for _ in range(5):
            self.mast_pubs[fid].publish(msg)
            time.sleep(0.04)

    def send_place(self, fid: int):
        """Command place-down and backout."""
        self.wait_for_subscribers(self.place_pubs[fid], timeout=1.5)
        for _ in range(5):
            self.place_pubs[fid].publish(Empty())
            time.sleep(0.04)

    def send_stop(self, fid: int):
        """Command emergency stop."""
        for _ in range(5):
            self.stop_pubs[fid].publish(Empty())
            time.sleep(0.04)

    def clear_status(self, fid: int):
        self.status_history[fid].clear()

    def wait_for_initial_data(self, timeout: float = 2.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if len(self.forklift_poses) >= len(FORKLIFT_IDS) and len(self.pallet_poses) >= len(PALLET_IDS):
                break
            time.sleep(0.05)


def wait_for_status(node: RLTesterNode, fid: int, prefix: str, timeout_sec: float = 60.0, target_xy=None, timeout: float = None, corridor_info=None):
    """
    Wait until forklift status matches prefix, or reports failure/rejection.
    Streams live telemetry (position, distance to goal, speed, steering) every second.
    If corridor_info=(pallet_xy, pallet_entry_heading) is provided, displays pocket corridor alignment errors.
    """
    if timeout is not None:
        timeout_sec = timeout
    t0 = time.time()
    last_print_t = 0.0
    last_pos = None
    stall_start_t = None

    try:
        while time.time() - t0 < timeout_sec:
            # Check received status history
            history = node.status_history.get(fid, [])
            for item in reversed(history):
                text, ts = item
                if ts >= t0:
                    if text.startswith(prefix):
                        dt = ts - t0
                        print(f"  {C_GREEN}✓ [{dt:5.1f} s]{C_RESET} {text}")
                        return True, text
                    elif text.startswith("failed") or text.startswith("rejected"):
                        dt = ts - t0
                        print(f"  {C_RED}✗ [{dt:5.1f} s]{C_RESET} {text}")
                        return False, text

            # Live telemetry printout every 1.0 s
            now = time.time()
            if now - last_print_t >= 1.0:
                last_print_t = now
                dt = now - t0
                p = node.forklift_poses.get(fid)
                ack = node.forklift_ackermann.get(fid)

                p_str = ""
                if p and (now - p[3] < 3.0):
                    dist_str = ""
                    if target_xy is not None:
                        dist_px = math.hypot(target_xy[0] - p[0], target_xy[1] - p[1])
                        dist_m = dist_px / PX_PER_M
                        dist_str = f" dist={dist_px:4.1f}px ({dist_m:.2f}m)"

                    hdg_deg = math.degrees(p[2])
                    corr_str = ""
                    if corridor_info is not None:
                        try:
                            pal_xy, pal_head = corridor_info
                            d_par, e_lat, e_yaw = pocket_corridor_errors(pal_xy, pal_head, (p[0], p[1]), p[2], px_per_m=PX_PER_M)
                            corr_str = f" corr=(d_par={d_par:4.2f}m, e_lat={e_lat:+5.2f}m, e_yaw={e_yaw:+5.1f}°)"
                        except Exception:
                            pass
                    p_str = f"pos=({p[0]:5.1f}, {p[1]:5.1f}) forks={hdg_deg:+5.1f}°{dist_str}{corr_str}"

                    # Stall detection: non-zero speed commanded but position unchanged for > 3 s
                    if last_pos is not None:
                        d_pos = math.hypot(p[0] - last_pos[0], p[1] - last_pos[1])
                        if d_pos < 2.0 and ack and abs(ack[0]) > 20.0:
                            if stall_start_t is None:
                                stall_start_t = now
                            elif now - stall_start_t > 3.0:
                                p_str += f" {C_YELLOW}[STALLED / OBSTRUCTED!]{C_RESET}"
                        else:
                            stall_start_t = None
                    last_pos = (p[0], p[1])

                cmd_str = ""
                if ack and (now - ack[2] < 2.0):
                    cmd_str = f" cmd=(spd={ack[0]:+5.0f}, str={ack[1]:+4.0f}°)"

                if p_str or cmd_str:
                    print(f"    {C_GRAY}[t+{dt:4.1f}s]{C_RESET} FL{fid}: {p_str}{cmd_str}")

            time.sleep(0.1)

    except KeyboardInterrupt:
        print(f"\n  {C_YELLOW}[INTERRUPTED]{C_RESET} Wait aborted by user.")
        return False, "interrupted"

    print(f"  {C_RED}✗ TIMEOUT{C_RESET} waiting for '{prefix}' after {timeout_sec:.1f} s")
    return False, "timeout"


def check_status(node: RLTesterNode):
    """Print preflight check of camera, forklifts, pallets, and lidar."""
    now = time.time()
    print(f"\n{C_BOLD}=== Swarm Status & Preflight Check ==={C_RESET}")

    print(f"\n{C_CYAN}Forklifts (from overhead camera):{C_RESET}")
    for fid in FORKLIFT_IDS:
        p = node.forklift_poses.get(fid)
        if p and (now - p[3] < 3.0):
            forks_deg = math.degrees(p[2])
            age = now - p[3]
            lidar_str = f"{C_GREEN}active{C_RESET}" if node.lidar_active.get(fid) else f"{C_YELLOW}none (run attach_fork_lidar.py){C_RESET}"
            print(f"  Forklift {fid}: px=({p[0]:6.1f}, {p[1]:6.1f})  forks={forks_deg:+6.1f}°  lidar: {lidar_str}  ({age:.1f}s ago)")
        else:
            print(f"  Forklift {fid}: {C_YELLOW}not seen by camera{C_RESET}")

    print(f"\n{C_CYAN}Pallets (from overhead camera):{C_RESET}")
    for pid in PALLET_IDS:
        p = node.pallet_poses.get(pid)
        if p and (now - p[3] < 3.0):
            heading_deg = math.degrees(p[2])
            age = now - p[3]
            print(f"  Pallet {pid}:   px=({p[0]:6.1f}, {p[1]:6.1f})  normal={heading_deg:+6.1f}°  ({age:.1f}s ago)")
        else:
            print(f"  Pallet {pid}:   {C_GRAY}not seen by camera{C_RESET}")

    print(f"\n{C_CYAN}Latest Goal Statuses:{C_RESET}")
    for fid in FORKLIFT_IDS:
        st = node.last_status.get(fid)
        if st and (now - st[1] < 30.0):
            print(f"  Forklift {fid}: {st[0]} ({now - st[1]:.1f}s ago)")
        else:
            print(f"  Forklift {fid}: (idle / no status yet)")
    print()


def test_nav(node: RLTesterNode, fid: int, x: float, y: float, heading: float = math.nan, timeout: float = 60.0):
    """Test the Go-To-Pose Nav RL policy."""
    h_str = f"{heading:.1f}°" if not math.isnan(heading) else "any"
    print(f"\n{C_BOLD}[TEST NAV]{C_RESET} Sending Forklift {fid} -> ({x:.1f}, {y:.1f}), heading: {h_str}")
    node.clear_status(fid)
    node.send_drive(fid, x, y, heading)
    ok, final = wait_for_status(node, fid, "reached:", timeout_sec=timeout, target_xy=(x, y))
    if ok:
        print(f"{C_GREEN}{C_BOLD}[PASS]{C_RESET} Nav policy reached target.")
    else:
        print(f"{C_RED}{C_BOLD}[FAIL]{C_RESET} Nav test failed: {final}")
    return ok


def test_pick(node: RLTesterNode, pallet_id: int, fid: int, lift_m: float = 0.3, timeout: float = 80.0):
    """Test Pallet Staging + Insert RL Policy."""
    now = time.time()
    p_pal = node.pallet_poses.get(pallet_id)
    p_fl = node.forklift_poses.get(fid)

    if not p_pal or (now - p_pal[3] > 5.0):
        print(f"{C_RED}[ERROR]{C_RESET} Pallet {pallet_id} is not visible to the camera!")
        return False
    if not p_fl or (now - p_fl[3] > 5.0):
        print(f"{C_RED}[ERROR]{C_RESET} Forklift {fid} is not visible to the camera!")
        return False

    # Check fork lidar
    if not node.lidar_active.get(fid):
        print(f"  {C_YELLOW}[WARN]{C_RESET} /forklift_{fid}/scan has no active data.")
        print(f"         Step 1 (Staging Nav) will proceed, but Step 2 (Insert) requires the fork lidar.")
        print(f"         In Isaac Sim: Window -> Script Editor -> File -> Open 'attach_fork_lidar.py' -> Run.")

    # Calculate staging pose
    heading = pallet_entry_heading(p_pal[2], (p_fl[0], p_fl[1]), (p_pal[0], p_pal[1]))
    sx, sy = staging_tag_pixels((p_pal[0], p_pal[1]), heading, px_per_m=PX_PER_M)
    h_deg = math.degrees(heading)

    print(f"\n{C_BOLD}[TEST PICK]{C_RESET} Forklift {fid} picking Pallet {pallet_id}:")
    print(f"  Staging target: ({sx:.1f}, {sy:.1f}), heading {h_deg:.1f}°")

    node.clear_status(fid)
    node.send_drive(fid, sx, sy, h_deg)

    print("  Step 1: Navigating to staging pose...")
    ok, st = wait_for_status(
        node, fid, "reached: staging",
        timeout_sec=50.0, target_xy=(sx, sy),
        corridor_info=((p_pal[0], p_pal[1]), heading)
    )
    if not ok:
        print(f"{C_RED}{C_BOLD}[FAIL]{C_RESET} Staging nav failed: {st}")
        return False

    print("  Step 2: Insert policy driving forks into pockets...")
    ok, st = wait_for_status(node, fid, "reached: inserted", timeout_sec=30.0)
    if not ok:
        if not node.lidar_active.get(fid):
            print(f"{C_RED}{C_BOLD}[FAIL]{C_RESET} Insert held because /forklift_{fid}/scan is inactive. Run attach_fork_lidar.py in Isaac Sim!")
        else:
            print(f"{C_RED}{C_BOLD}[FAIL]{C_RESET} Insert failed: {st}")
        return False

    print(f"  Step 3: Raising mast to {lift_m:.2f} m carry height...")
    node.send_mast(fid, lift_m)
    time.sleep(2.5)

    print(f"{C_GREEN}{C_BOLD}[PASS]{C_RESET} Pick successful! Pallet is on forks.")
    return True


def test_place(node: RLTesterNode, fid: int, timeout: float = 30.0):
    """Test Place & Backout."""
    print(f"\n{C_BOLD}[TEST PLACE]{C_RESET} Commanding Forklift {fid} to set down pallet and back out...")
    node.clear_status(fid)
    node.send_place(fid)
    ok, st = wait_for_status(node, fid, "reached: placed", timeout_sec=timeout)
    if ok:
        print(f"{C_GREEN}{C_BOLD}[PASS]{C_RESET} Pallet placed and backed out cleanly.")
    else:
        print(f"{C_RED}{C_BOLD}[FAIL]{C_RESET} Place failed: {st}")
    return ok


def test_cycle(node: RLTesterNode, pallet_id: int, fid: int, drop_x: float = 900.0, drop_y: float = 500.0, drop_h: float = 90.0):
    """Full End-to-End Test: Stage -> Insert -> Lift -> Deliver -> Place."""
    print(f"\n{C_BOLD}===================================================={C_RESET}")
    print(f"{C_BOLD}   Full End-to-End Cycle Test (Forklift {fid}, Pallet {pallet_id})   {C_RESET}")
    print(f"{C_BOLD}===================================================={C_RESET}")

    t_start = time.time()

    # 1. Pick
    if not test_pick(node, pallet_id, fid, lift_m=0.3):
        return False

    # 2. Deliver
    print(f"\n  Step 4: Delivering carried pallet to ({drop_x:.1f}, {drop_y:.1f}, {drop_h:.1f}°)...")
    node.clear_status(fid)
    node.send_drive(fid, drop_x, drop_y, drop_h)
    ok, st = wait_for_status(node, fid, "reached: delivered", timeout_sec=50.0, target_xy=(drop_x, drop_y))
    if not ok:
        print(f"{C_RED}{C_BOLD}[FAIL]{C_RESET} Delivery travel failed: {st}")
        return False

    # 3. Place
    if not test_place(node, fid):
        return False

    total_t = time.time() - t_start
    print(f"\n{C_GREEN}{C_BOLD}★ FULL CYCLE COMPLETED SUCCESSFULLY in {total_t:.1f} s! ★{C_RESET}\n")
    return True


def live_monitor(node: RLTesterNode):
    """Stream live dashboard of the swarm."""
    print(f"{C_CYAN}Streaming live swarm status. Press Ctrl+C to return to menu.{C_RESET}\n")
    try:
        while True:
            os.system('clear')
            print(f"{C_BOLD}Live Swarm Monitor (Press Ctrl+C to exit){C_RESET}")
            check_status(node)
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


def execute_fleet_command(node: RLTesterNode, line: str) -> bool:
    """Execute raw fleet commands matching fleet_controller.py syntax."""
    parts = line.strip().split()
    if not parts:
        return True

    cmd = parts[0].lower()

    if cmd in ('help', 'h', '?'):
        print_help()
        return True

    if cmd in ('quit', 'exit', 'q'):
        return False

    if cmd == 'status':
        check_status(node)
        return True

    if cmd == 'monitor':
        live_monitor(node)
        return True

    if cmd == 'stop':
        if len(parts) == 1 or parts[1].lower() == 'all':
            for fid in FORKLIFT_IDS:
                node.send_stop(fid)
            print("  All forklifts stopped.")
        else:
            fid = int(parts[1])
            node.send_stop(fid)
            print(f"  Forklift {fid} stopped.")
        return True

    if cmd == 'place':
        if len(parts) < 2:
            print("  Usage: place <forklift_id>")
            return True
        test_place(node, int(parts[1]))
        return True

    if cmd == 'lift':
        if len(parts) < 3:
            print("  Usage: lift <forklift_id> <height_m>")
            return True
        node.send_mast(int(parts[1]), float(parts[2]))
        print(f"  Forklift {parts[1]} mast commanded to {parts[2]} m.")
        return True

    if cmd == 'pallet':
        if len(parts) not in (3, 4):
            print("  Usage: pallet <pallet_id> <forklift_id> [lift_m]")
            return True
        pallet_id = int(parts[1])
        fid = int(parts[2])
        lift = float(parts[3]) if len(parts) == 4 else 0.3
        test_pick(node, pallet_id, fid, lift)
        return True

    if cmd == 'cycle':
        if len(parts) not in (3, 6):
            print("  Usage: cycle <pallet_id> <forklift_id> [drop_x drop_y drop_h]")
            return True
        pallet_id = int(parts[1])
        fid = int(parts[2])
        if len(parts) == 6:
            test_cycle(node, pallet_id, fid, float(parts[3]), float(parts[4]), float(parts[5]))
        else:
            test_cycle(node, pallet_id, fid)
        return True

    # Try format: <forklift_id> <x> <y> [heading_deg|-] [lift_m]
    try:
        fid = int(parts[0])
        if fid in FORKLIFT_IDS and len(parts) >= 3:
            x = float(parts[1])
            y = float(parts[2])
            h = math.nan
            if len(parts) >= 4 and parts[3] != '-':
                h = float(parts[3])
            test_nav(node, fid, x, y, h)
            if len(parts) >= 5:
                node.send_mast(fid, float(parts[4]))
            return True
    except ValueError:
        pass

    print(f"  {C_RED}[ERROR]{C_RESET} Unknown command: '{line}'. Type 'help' or select a menu number.")
    return True


def print_help():
    print(f"""
{C_BOLD}Commands Supported (Direct Fleet Mode & Shortcuts):{C_RESET}
  {C_CYAN}pallet <pallet> <forklift> [lift_m]{C_RESET}   Pick pallet (stage -> insert -> lift to lift_m, default 0.3m)
  {C_CYAN}<forklift> <x> <y> [heading|-] [lift]{C_RESET} Drive forklift to camera pixels (x, y) with optional heading
  {C_CYAN}place <forklift>{C_RESET}                       Set down carried pallet and back out
  {C_CYAN}lift <forklift> <height_m>{C_RESET}            Command lift mast height
  {C_CYAN}stop [all|forklift]{C_RESET}                  Emergency stop
  {C_CYAN}cycle <pallet> <forklift>{C_RESET}              Run complete cycle (pick -> deliver -> place)
  {C_CYAN}status{C_RESET}                                Show latest poses of all forklifts & pallets
  {C_CYAN}monitor{C_RESET}                               Open live 1-Hz streaming dashboard
  {C_CYAN}1-7{C_RESET}                                    Numbered interactive test wizard
  {C_CYAN}help / quit{C_RESET}                            Show help or exit tester

{C_BOLD}Examples:{C_RESET}
  pallet 1 3 0.3      -> Forklift 3 picks pallet 1, inserts, and lifts to 0.3 m
  3 550 500 180       -> Forklift 3 navigates to pixel (550, 500), forks facing 180°
  place 3             -> Forklift 3 places down the pallet and backs out
  stop all            -> Stop all forklifts immediately
""")


def interactive_menu(node: RLTesterNode):
    """Unified interactive prompt: accepts both menu choices and direct fleet commands."""
    print(f"\n{C_BOLD}=========================================================={C_RESET}")
    print(f"{C_BOLD}         Centralized Forklift RL Swarm Tester             {C_RESET}")
    print(f"{C_BOLD}=========================================================={C_RESET}")
    print("  You can type direct fleet commands or select a menu option:")
    print("    • pallet <P> <F> [lift]   (e.g. pallet 1 3 0.3)")
    print("    • <F> <x> <y> [heading]   (e.g. 3 550 500 180)")
    print("    • place <F>               (e.g. place 3)")
    print("    • stop all / stop <F>")
    print("  Menu Options:")
    print("    1) Preflight Check & Swarm Status")
    print("    2) Test Nav Policy (Go-to-Pose)")
    print("    3) Test Insert Policy (Pick Pallet: Stage -> Insert -> Lift)")
    print("    4) Test Place Policy (Set down & Back out)")
    print("    5) Run Full Cycle (Pick -> Deliver -> Place)")
    print("    6) Live Dashboard Monitor")
    print("    7) Emergency Stop All Forklifts")
    print("    h) Help / Syntax Guide")
    print("    q) Quit")
    print("----------------------------------------------------------")

    while True:
        try:
            line = input(f"{C_CYAN}tester> {C_RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if not line:
            continue

        if line in ('q', 'quit', 'exit'):
            break

        # Numbered menu choices
        if line == '1':
            check_status(node)
        elif line == '2':
            try:
                fid = int(input("  Forklift ID [1, 2, or 3] (default 3): ").strip() or "3")
                x = float(input("  Target X (pixels, e.g. 550): ").strip() or "550")
                y = float(input("  Target Y (pixels, e.g. 500): ").strip() or "500")
                h_in = input("  Target Heading deg [0 right, 90 up, 180 left, -90 down, or '-' for any]: ").strip()
                h = float(h_in) if h_in and h_in != '-' else math.nan
                test_nav(node, fid, x, y, h)
            except ValueError as e:
                print(f"  Invalid input: {e}")
        elif line == '3':
            try:
                pal = int(input("  Pallet ID [1, 2, or 3] (default 1): ").strip() or "1")
                fid = int(input("  Forklift ID [1, 2, or 3] (default 3): ").strip() or "3")
                lift = float(input("  Carry lift height in meters (default 0.3): ").strip() or "0.3")
                test_pick(node, pal, fid, lift)
            except ValueError as e:
                print(f"  Invalid input: {e}")
        elif line == '4':
            try:
                fid = int(input("  Forklift ID [1, 2, or 3] (default 3): ").strip() or "3")
                test_place(node, fid)
            except ValueError as e:
                print(f"  Invalid input: {e}")
        elif line == '5':
            try:
                pal = int(input("  Pallet ID [1, 2, or 3] (default 1): ").strip() or "1")
                fid = int(input("  Forklift ID [1, 2, or 3] (default 3): ").strip() or "3")
                test_cycle(node, pal, fid)
            except ValueError as e:
                print(f"  Invalid input: {e}")
        elif line == '6':
            live_monitor(node)
        elif line == '7':
            print("Stopping all forklifts...")
            for fid in FORKLIFT_IDS:
                node.send_stop(fid)
            print("Done.")
        elif line in ('h', 'help', '?'):
            print_help()
        else:
            # Pass to fleet command parser
            if not execute_fleet_command(node, line):
                break


def main():
    parser = argparse.ArgumentParser(description="Test Forklift RL Policies")
    parser.add_argument("--check", action="store_true", help="Check status and exit")
    parser.add_argument("--monitor", action="store_true", help="Live stream status")
    parser.add_argument("--nav", nargs="+", metavar="ARG", help="--nav <forklift_id> <x> <y> [heading_deg]")
    parser.add_argument("--pick", nargs="+", metavar="ARG", help="--pick <pallet_id> <forklift_id> [lift_m]")
    parser.add_argument("--place", type=int, metavar="FID", help="--place <forklift_id>")
    parser.add_argument("--cycle", nargs="+", metavar="ARG", help="--cycle <pallet_id> <forklift_id>")
    parser.add_argument("--stop", nargs="?", const="all", help="--stop [all|1|2|3]")

    args = parser.parse_args()

    rclpy.init()
    node = RLTesterNode()

    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()

    # Give subscriptions a moment to discover publishers & gather camera data
    node.wait_for_initial_data(timeout=2.5)

    try:
        if args.check:
            check_status(node)
        elif args.monitor:
            live_monitor(node)
        elif args.nav:
            fid = int(args.nav[0])
            x, y = float(args.nav[1]), float(args.nav[2])
            h = float(args.nav[3]) if len(args.nav) > 3 and args.nav[3] != '-' else math.nan
            test_nav(node, fid, x, y, h)
        elif args.pick:
            pal = int(args.pick[0])
            fid = int(args.pick[1])
            lift = float(args.pick[2]) if len(args.pick) > 2 else 0.3
            test_pick(node, pal, fid, lift)
        elif args.place is not None:
            test_place(node, args.place)
        elif args.cycle:
            pal = int(args.cycle[0])
            fid = int(args.cycle[1])
            test_cycle(node, pal, fid)
        elif args.stop:
            if args.stop.lower() == 'all':
                for fid in FORKLIFT_IDS:
                    node.send_stop(fid)
                print("All forklifts stopped.")
            else:
                fid = int(args.stop)
                node.send_stop(fid)
                print(f"Forklift {fid} stopped.")
        else:
            interactive_menu(node)
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        thread.join(timeout=0.5)
        try:
            node.destroy_node()
        except Exception:
            pass


if __name__ == '__main__':
    main()
