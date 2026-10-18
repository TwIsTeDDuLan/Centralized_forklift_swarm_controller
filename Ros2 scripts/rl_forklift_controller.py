#!/usr/bin/env python3
"""
RL go-to-pose controller: drop-in replacement for the drive part of forklift_controller.py.

Same topics as the hand-tuned controller, so the fleet / task-allocation layer does not change:
  in   /forklift_N/vision_data      (Pose2D: tag x, y [px], theta [rad])   from vision_tracker.py
  in   /forklift_N/lift_target_pos  (Pose2D: x, y [px], theta = final fork heading [deg] or NaN)
  in   /forklift_N/mast_target_pos, /forklift_N/stop                       (handled by ForkliftController)
  out  /forklift_N/ackermann        (speed [deg/s], steering_angle [deg])
  out  /forklift_N/goal_status      ("reached: ..." / "failed: ...")

The policy was trained in Isaac Lab (isaac_lab_forklift/) in meters. This node converts the camera
pixels to meters (px_per_m), builds the same observation vector as ForkliftNavEnv and runs the exported
actor MLP in numpy (policies/forklift_nav.npz, written by isaac_lab_forklift/scripts/export_policy.py).
Its actions are already in Ackermann topic units, so no drive_sign / steering_sign calibration is needed.

    python3 rl_forklift_controller.py 1        # forklift 1;  or ./run_rl_forklift.sh 1
"""

import math
import os
import sys

import numpy as np

from forklift_controller import ForkliftController, clamp, wrap
from rl_policy import NumpyPolicy, build_actor_obs

import rclpy
from geometry_msgs.msg import Pose2D

RL = 'rl'
DEFAULT_POLICY = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'policies', 'forklift_nav.npz')


class RLForkliftController(ForkliftController):
    def __init__(self, forklift_id: int):
        super().__init__(forklift_id)
        p = self.declare_parameter
        policy_path = p('policy_path', DEFAULT_POLICY).value
        # Overhead camera scale at the tag height (camera 20 m up, tag ~2.95 m above the floor).
        self.px_per_m = p('px_per_m', 65.0).value
        # Consecutive control ticks inside tolerance before the goal counts as reached.
        self.settle_ticks = p('settle_ticks', 3).value
        self.auto_calibrate = False

        self.policy = NumpyPolicy(policy_path)
        meta = self.policy.meta
        self.max_speed = meta['max_wheel_speed_deg']
        self.max_steering = meta['max_steer_deg']
        # Tolerances default to the ones the policy was trained to hit (meters -> pixels).
        self.position_tolerance = p('rl_position_tolerance_px', meta['pos_tolerance'] * self.px_per_m * 1.25).value
        self.heading_tolerance = math.radians(
            p('rl_heading_tolerance_deg', math.degrees(meta['heading_tolerance']) * 1.5).value)
        if abs(meta['control_dt'] - 0.1) > 1e-6:
            self.get_logger().warn(f"policy trained at dt={meta['control_dt']} s but this node runs at 0.1 s")

        self.last_action = np.zeros(2)
        self.inside_ticks = 0
        self.get_logger().info(f"RL policy loaded: {policy_path} ({meta['checkpoint']})")

    # ------Frames------
    def to_meters(self, x_px: float, y_px: float):
        """Image pixels (y down) -> meters in a y-up frame (same handedness as the tracker's angles)."""
        return x_px / self.px_per_m, -y_px / self.px_per_m

    def body_yaw(self) -> float:
        """Yaw of the body +X axis (counterweight side) = fork heading - 180 deg."""
        return wrap(self.heading() - math.pi)

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
        self.state = RL
        self.goal_start_time = self.now_sec()
        self.last_action[:] = 0.0
        self.inside_ticks = 0
        heading_txt = 'any' if self.goal_heading is None else f"{msg.theta:.1f}°"
        self.get_logger().info(f"New goal (RL): x={msg.x:.1f}, y={msg.y:.1f}, heading={heading_txt}")

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
                self.finish(True, detail)
                return
        else:
            self.inside_ticks = 0

        goal_yaw = None if self.goal_heading is None else wrap(self.goal_heading - math.pi)
        obs = build_actor_obs(self.policy.meta, self.to_meters(self.cur_pos.x, self.cur_pos.y), self.body_yaw(),
                              self.to_meters(self.goal.x, self.goal.y), goal_yaw, self.last_action)
        action = self.policy(obs)
        self.last_action = action
        speed = clamp(action[0] * self.max_speed, -self.max_speed, self.max_speed)
        steering = clamp(action[1] * self.max_steering, -self.max_steering, self.max_steering)
        self.publish_drive(speed, steering)

        self.get_logger().info(
            f"[rl] pos=({self.cur_pos.x:.0f}, {self.cur_pos.y:.0f}) heading={math.degrees(self.heading()):.0f}° "
            f"{detail} cmd speed={speed:.0f} steer={steering:.0f}",
            throttle_duration_sec=1.0
        )


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
