#!/usr/bin/env python3
"""
Two-policy RL controller: nav MLP to a staging pose, then insert MLP into the pallet pockets.

Pallet cycle (fleet_controller.py `pallet` / `place` do this for you):
  1. lift_target_pos = staging tag pose -> nav policy drives there ("reached: staging")
  2. insert policy drives the forks into the pockets         ("reached: inserted ...")
  3. mast_target_pos z = carry height                         (lifts the pallet)
  4. lift_target_pos = destination; while carrying, no insert ("reached: delivered ...")
  5. /forklift_N/place (Empty): lower the forks, back straight out ("reached: placed ...")

Same topics as the hand-tuned controller, plus an optional /forklift_N/scan (LaserScan, 192 ranges):
  in   /forklift_N/vision_data      (Pose2D: tag x, y [px], theta [rad])
  in   /forklift_N/lift_target_pos  (Pose2D: staging tag x, y [px], theta = fork heading [deg] or NaN)
  in   /forklift_N/mast_target_pos, /forklift_N/stop, /forklift_N/scan, /forklift_N/place
  out  /forklift_N/ackermann, /forklift_N/lift, /forklift_N/goal_status

    python3 rl_forklift_controller.py 1
"""

import math
import os
import sys

import numpy as np

from forklift_controller import ForkliftController, clamp, wrap
from rl_policy import NumpyPolicy, build_actor_obs, build_insert_obs, lift_joint_norm

import rclpy
from geometry_msgs.msg import Pose2D, Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Empty

NAV_STAGING = 'nav_staging'
INSERT = 'insert'
PLACE = 'place'
DEFAULT_NAV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'policies', 'forklift_nav.npz')
DEFAULT_INSERT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'policies', 'forklift_insert.npz')


class RLForkliftController(ForkliftController):
    def __init__(self, forklift_id: int):
        super().__init__(forklift_id)
        p = self.declare_parameter
        nav_path = p('policy_path', DEFAULT_NAV).value
        insert_path = p('insert_policy_path', DEFAULT_INSERT).value
        self.px_per_m = p('px_per_m', 65.0).value
        self.settle_ticks = p('settle_ticks', 3).value
        self.auto_calibrate = False
        self.heading_offset = math.radians(p('tag_heading_offset_deg', 0.0).value)
        self.auto_heading_check = p('auto_heading_check', True).value
        self.heading_check = {'window': None, 'votes': [], 'locked': False}
        self.auto_insert = p('auto_insert', True).value
        self.insert_progress_m = p('insert_progress_m', 2.0).value
        self.insert_timeout = p('insert_timeout_sec', 20.0).value

        self.nav_policy = NumpyPolicy(nav_path)
        nav_meta = self.nav_policy.meta
        self.max_speed = nav_meta['max_wheel_speed_deg']
        self.max_steering = nav_meta['max_steer_deg']
        # Staging is looser than nav training (0.5 m, ~30 deg) per the two-policy plan.
        self.position_tolerance = p('rl_position_tolerance_px', 0.5 * self.px_per_m).value
        self.heading_tolerance = math.radians(p('rl_heading_tolerance_deg', 30.0).value)

        self.insert_policy = None
        self.max_lift_speed = 0.30
        if os.path.isfile(insert_path):
            self.insert_policy = NumpyPolicy(insert_path)
            self.max_lift_speed = self.insert_policy.meta.get('max_lift_speed', 0.30)
            self.get_logger().info(f"Insert policy loaded: {insert_path} ({self.insert_policy.meta.get('checkpoint')})")
        else:
            self.get_logger().warn(f"No insert policy at {insert_path}; auto_insert will stop at staging.")
            self.auto_insert = False

        self.last_action = np.zeros(2)
        self.last_insert_action = np.zeros(3)
        self.inside_ticks = 0
        self._policy_owns_lift = False
        self._scan = None
        self._scan_time = None
        self._staging_start = None
        self._insert_start = None
        # Pallet on the forks: nav goals end at the goal (no insert) until /place.
        self.carrying = False
        self.insert_lift_height = None  # mast height when the forks went into the pockets
        self._place = None
        self.place_lower_m = p('place_lower_m', 0.05).value  # below the insert height when setting down
        self.place_backout_m = p('place_backout_m', 2.5).value
        self.place_timeout = p('place_timeout_sec', 20.0).value
        ns = f'/forklift_{self.fid}'
        self.create_subscription(LaserScan, f'{ns}/scan', self.scan_callback, 10)
        self.create_subscription(Empty, f'{ns}/place', self.place_callback, 10)
        self.get_logger().info(f"RL nav policy loaded: {nav_path} ({nav_meta['checkpoint']})")

    # ------Frames------
    def to_meters(self, x_px: float, y_px: float):
        return x_px / self.px_per_m, -y_px / self.px_per_m

    def body_yaw(self) -> float:
        return wrap(self.heading() - math.pi)

    def scan_callback(self, msg: LaserScan):
        self._scan = np.asarray(msg.ranges, dtype=np.float64)
        self._scan_time = self.now_sec()

    # ------Goals------
    def goal_callback(self, msg: Pose2D):
        if not self.in_safe_area(msg.x, msg.y):
            text = (f"rejected: target ({msg.x:.0f}, {msg.y:.0f}) is outside the safe camera area "
                    f"x[{self.safe_margin}, {self.camera_width - self.safe_margin}] "
                    f"y[{self.safe_margin}, {self.camera_height - self.safe_margin}]")
            self.get_logger().warn(text)
            self.publish_status(text)
            return
        self.goal = Pose2D(x=float(msg.x), y=float(msg.y), theta=float(msg.theta))
        self.goal_heading = None if math.isnan(msg.theta) else math.radians(msg.theta)
        self.state = NAV_STAGING
        self.goal_start_time = self.now_sec()
        self.last_action[:] = 0.0
        self.last_insert_action[:] = 0.0
        self.inside_ticks = 0
        self._policy_owns_lift = False
        self._staging_start = None
        heading_txt = 'any' if self.goal_heading is None else f"{msg.theta:.1f}°"
        kind = 'delivery goal (carrying, no insert)' if self.carrying else 'staging goal'
        self.get_logger().info(f"New {kind} (RL): x={msg.x:.1f}, y={msg.y:.1f}, heading={heading_txt}")

    def place_callback(self, msg: Empty):
        if self.state not in ('idle',):
            self.get_logger().warn("Place ignored: finish or /stop the current goal first.")
            self.publish_status("failed: place ignored, forklift busy")
            return
        if not self.carrying:
            self.get_logger().warn("Place requested while not carrying; backing out anyway.")
        lower_to = (self.insert_lift_height if self.insert_lift_height is not None else self.mast_height)
        lower_to = clamp(lower_to - self.place_lower_m, -0.15, 2.0)
        self.state = PLACE
        self._policy_owns_lift = True
        self._place = {'phase': 'lower', 'lift': lower_to, 'start': self.now_sec(), 'from': None}
        self.get_logger().info(f"Placing: lowering forks to {lower_to:.2f} m, then backing out "
                               f"{self.place_backout_m:.1f} m.")

    def mast_control_loop(self):
        if self._policy_owns_lift:
            return
        super().mast_control_loop()

    def finish(self, success: bool, reason: str):
        self._policy_owns_lift = False
        super().finish(success, reason)

    def publish_lift(self, height: float):
        cmd = Twist()
        cmd.linear.z = float(clamp(height, -0.15, 2.0))
        self.pub_lift.publish(cmd)

    # ------Drive------
    def drive_control_loop(self):
        if self.startup_stop_ticks > 0:
            self.startup_stop_ticks -= 1
            self.publish_stop()
            return
        if self.state == 'idle':
            return

        now = self.now_sec()
        if self.last_vision_time is None or now - self.last_vision_time > self.vision_timeout:
            self.publish_stop()
            self.last_action[:] = 0.0
            self.get_logger().warn("No fresh vision data for this forklift; holding until the camera sees it again.",
                                   throttle_duration_sec=2.0)
            return

        if self.state == NAV_STAGING:
            self._nav_tick(now)
        elif self.state == INSERT:
            self._insert_tick(now)
        elif self.state == PLACE:
            self._place_tick(now)

    def _nav_tick(self, now: float):
        if now - self.goal_start_time > self.goal_timeout:
            self.finish(False, f"timed out after {self.goal_timeout:.0f} s")
            return
        dist_px = math.hypot(self.goal.x - self.cur_pos.x, self.goal.y - self.cur_pos.y)
        heading_error = None if self.goal_heading is None else wrap(self.goal_heading - self.heading())
        heading_ok = heading_error is None or abs(heading_error) <= self.heading_tolerance
        detail = f"position error {dist_px:.1f} px"
        if heading_error is not None:
            detail += f", heading error {math.degrees(heading_error):.1f}°"
        if dist_px < self.position_tolerance and heading_ok:
            self.inside_ticks += 1
            if self.inside_ticks >= self.settle_ticks:
                if self.carrying:
                    self.finish(True, f"delivered ({detail})")
                    return
                if self.auto_insert and self.insert_policy is not None:
                    self.state = INSERT
                    self._policy_owns_lift = True
                    self._insert_start = now
                    # insert progress is measured from where the insert starts, along the forks
                    self._staging_start = (self.cur_pos.x, self.cur_pos.y)
                    self.last_insert_action[:] = 0.0
                    self.publish_status("reached: staging")
                    self.get_logger().info(f"Staging reached ({detail}); switching to insert policy.")
                    return
                self.finish(True, f"staging ({detail})")
                return
        else:
            self.inside_ticks = 0

        goal_yaw = None if self.goal_heading is None else wrap(self.goal_heading - math.pi)
        obs = build_actor_obs(self.nav_policy.meta, self.to_meters(self.cur_pos.x, self.cur_pos.y), self.body_yaw(),
                              self.to_meters(self.goal.x, self.goal.y), goal_yaw, self.last_action)
        action = self.nav_policy(obs)
        self.last_action = action
        speed = clamp(action[0] * self.max_speed, -self.max_speed, self.max_speed)
        steering = clamp(action[1] * self.max_steering, -self.max_steering, self.max_steering)
        self.publish_drive(speed, steering)
        if self.auto_heading_check:
            self.check_heading_offset(speed, steering)
        self.get_logger().info(
            f"[nav] pos=({self.cur_pos.x:.0f}, {self.cur_pos.y:.0f}) heading={math.degrees(self.heading()):.0f}° "
            f"{detail} cmd speed={speed:.0f} steer={steering:.0f}",
            throttle_duration_sec=1.0
        )

    def _insert_tick(self, now: float):
        if now - (self._insert_start or now) > self.insert_timeout:
            self.finish(False, f"insert timed out after {self.insert_timeout:.0f} s")
            return
        if self._scan is None or self._scan_time is None or now - self._scan_time > 0.5:
            self.publish_stop()
            self.get_logger().warn("No fresh /scan for insert policy; holding.", throttle_duration_sec=2.0)
            return

        if self._staging_start is not None:
            dx = (self.cur_pos.x - self._staging_start[0]) / self.px_per_m
            dy = -(self.cur_pos.y - self._staging_start[1]) / self.px_per_m
            progress = dx * math.cos(self.heading()) + dy * math.sin(self.heading())
            if progress >= self.insert_progress_m:
                self.carrying = True
                self.insert_lift_height = self.mast_height
                # hold the lift where the insert left it until a mast_target_pos arrives
                self.mast_target_received = False
                self.publish_lift(self.mast_height)
                self.finish(True, f"inserted ({progress:.2f} m along forks from staging)")
                return

        n_rays = 192
        ranges = self._scan
        if ranges.size < n_rays:
            ranges = np.pad(ranges, (0, n_rays - ranges.size), constant_values=self.insert_policy.meta['lidar_max_range'])
        elif ranges.size > n_rays:
            ranges = ranges[:n_rays]
        obs = build_insert_obs(
            self.insert_policy.meta, ranges, self.last_insert_action, lift_joint_norm(self.mast_height)
        )
        action = self.insert_policy(obs)
        if action.shape[0] < 3:
            action = np.pad(action, (0, 3 - action.shape[0]))
        self.last_insert_action = action[:3]
        speed = clamp(action[0] * self.max_speed, -self.max_speed, self.max_speed)
        steering = clamp(action[1] * self.max_steering, -self.max_steering, self.max_steering)
        lift = clamp(self.mast_height + action[2] * self.max_lift_speed * 0.1, -0.15, 2.0)
        self.publish_drive(speed, steering)
        self.publish_lift(lift)
        self.get_logger().info(
            f"[insert] pos=({self.cur_pos.x:.0f}, {self.cur_pos.y:.0f}) lift={self.mast_height:.2f} "
            f"cmd speed={speed:.0f} steer={steering:.0f} lift={lift:.2f}",
            throttle_duration_sec=1.0
        )

    def _place_tick(self, now: float):
        pl = self._place
        if now - pl['start'] > self.place_timeout:
            self.finish(False, f"place timed out after {self.place_timeout:.0f} s")
            return
        if pl['phase'] == 'lower':
            self.publish_stop()
            self.publish_lift(pl['lift'])
            if abs(self.mast_height - pl['lift']) < 0.02 or now - pl['start'] > 6.0:
                pl['phase'] = 'backout'
                pl['from'] = (self.cur_pos.x, self.cur_pos.y)
            return
        # Back straight out, counterweight first (+speed drives body +X, away from the forks).
        dx = (self.cur_pos.x - pl['from'][0]) / self.px_per_m
        dy = -(self.cur_pos.y - pl['from'][1]) / self.px_per_m
        backed = -(dx * math.cos(self.heading()) + dy * math.sin(self.heading()))
        if backed >= self.place_backout_m:
            self.carrying = False
            self.insert_lift_height = None
            self.target_mast_height = pl['lift']  # do not let the mast PID jump back to the carry height
            self.finish(True, f"placed (backed out {backed:.2f} m)")
            return
        self.publish_lift(pl['lift'])
        self.publish_drive(0.5 * self.max_speed, 0.0)

    # ------Heading offset self-check------
    def check_heading_offset(self, speed: float, steering: float):
        """Compare the direction the tag actually moves with the direction the forklift believes it faces."""
        c = self.heading_check
        if c['locked']:
            return
        straight = abs(steering) < 20.0 and abs(speed) > 0.4 * self.max_speed
        key = speed > 0
        w = c['window']
        if not straight or w is None or w['key'] != key:
            c['window'] = {'key': key, 'x': self.cur_pos.x, 'y': self.cur_pos.y, 'yaw': self.body_yaw()} if straight else None
            return
        dx, dy = self.cur_pos.x - w['x'], self.cur_pos.y - w['y']
        if math.hypot(dx, dy) < 30.0:
            return
        c['window'] = None
        moved = math.atan2(-dy, dx)
        believed = wrap(w['yaw'] + (0.0 if key else math.pi))
        step = round(wrap(moved - believed) / (math.pi / 2)) % 4
        c['votes'].append(step)
        if len(c['votes']) < 2 or c['votes'][-1] != c['votes'][-2]:
            return
        if step == 0:
            c['locked'] = True
            self.get_logger().info("Heading offset confirmed by observed motion.")
            return
        self.heading_offset = wrap(self.heading_offset + step * math.pi / 2)
        offset_deg = round(math.degrees(self.heading_offset)) + 0
        c['votes'].clear()
        self.last_action[:] = 0.0
        self.get_logger().warn(
            f"Observed motion is {step * 90} deg off the tracked heading; tag_heading_offset_deg is now "
            f"{offset_deg}. Launch with '--ros-args -p tag_heading_offset_deg:={offset_deg}.0' to make it permanent.")


def main(forklift_id: int, args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = RLForkliftController(forklift_id)
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        if node is not None:
            if rclpy.ok():
                node.publish_stop()
            node.get_logger().info(f"Shutting down RL Forklift {forklift_id}...")
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    if len(sys.argv) < 2 or not sys.argv[1].isdigit():
        sys.exit("usage: rl_forklift_controller.py <forklift_id> [--ros-args ...]")
    main(int(sys.argv[1]), args=sys.argv[2:])
