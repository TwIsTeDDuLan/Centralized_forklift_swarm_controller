#!/usr/bin/env python3
"""
Go-to-pose controller shared by all forklift nodes (Fork Lifts/Fork_Lift_N.py).

Coordinates come from the overhead camera (vision_tracker.py):
  * x, y      : pixels, image origin top-left, y pointing down.
  * headings  : CCW from the image +x axis with image "up" = +90 deg.

Goal topic  /forklift_N/lift_target_pos (geometry_msgs/Pose2D)
  * x, y      : target pixel.
  * theta     : final heading in DEGREES (direction the forks point),
                or NaN to just drive to the point.

Isaac Sim drive model (Forklift_ROS2.usd action graph):
  * AckermannDrive.speed          -> rear wheel joint target velocity (deg/s).
  * AckermannDrive.steering_angle -> rear swivel joint target angle (deg, limit +-60).
The steered wheel is at the rear, so the yaw direction of a steering command
flips with the wheel's travel direction.
"""

import os
import sys

# Add ROS Python packages path if the shell has not sourced ROS setup.
ros_python_path = '/opt/ros/jazzy/lib/python3.12/site-packages'
if os.path.isdir(ros_python_path) and ros_python_path not in sys.path:
    sys.path.append(ros_python_path)

import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist, Pose2D
from std_msgs.msg import Empty, String
from ackermann_msgs.msg import AckermannDriveStamped

IDLE = 'idle'
APPROACH = 'approach'
FINAL = 'final'


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def image_angle(dx, dy):
    """Angle of an image-space vector using the tracker's convention (image y points down)."""
    return math.atan2(-dy, dx)


def clamp(value, low, high):
    return max(low, min(high, value))


class ForkliftController(Node):
    def __init__(self, forklift_id: int):
        super().__init__(f'forklift_{forklift_id}')
        self.fid = forklift_id
        ns = f'/forklift_{forklift_id}'

        # ------Parameters------
        p = self.declare_parameter
        # Angle from the ArUco tag's "front" edge to the direction the forks point.
        self.heading_offset = math.radians(p('heading_offset_deg', 180.0).value)
        # Wheel speed sign that drives the forklift forks-first.
        self.drive_sign = p('drive_sign', -1.0).value
        # Yaw direction = steering_sign * sign(wheel speed) * sign(steering angle).
        self.steering_sign = p('steering_sign', -1.0).value
        self.auto_calibrate = p('auto_calibrate_signs', True).value

        self.max_speed = p('max_speed', 240.0).value          # deg/s of the drive wheel
        self.min_speed = p('min_speed', 40.0).value
        self.speed_gain = p('speed_gain', 1.5).value          # (deg/s) per pixel of remaining distance
        self.max_steering = p('max_steering_deg', 55.0).value

        # Geometry in camera pixels (forklift_b: tag 1.2 m and rear wheel 1.39 m behind the
        # front rollers, ~65 px/m with the overhead camera at 20 m).
        self.tag_offset = p('tag_offset_px', 78.0).value
        self.wheelbase = p('wheelbase_px', 90.0).value
        # Virtual point ahead of the front rollers that is steered when driving forks-first.
        self.lead_offset = p('lead_offset_px', 60.0).value

        self.position_tolerance = p('position_tolerance_px', 15.0).value
        # Accepted error when the forklift crosses the goal at the end of a heading-constrained approach.
        self.final_tolerance = p('final_tolerance_px', 30.0).value
        self.heading_tolerance = math.radians(p('heading_tolerance_deg', 20.0).value)
        self.approach_distance = p('approach_distance_px', 200.0).value
        self.approach_tolerance = p('approach_tolerance_px', 25.0).value
        self.lookahead = p('lookahead_px', 60.0).value
        self.goal_timeout = p('goal_timeout_sec', 120.0).value
        self.vision_timeout = p('vision_timeout_sec', 0.5).value
        self.max_final_retries = p('max_final_retries', 2).value

        self.camera_width = p('camera_width', 1280).value
        self.camera_height = p('camera_height', 720).value
        self.safe_margin = p('safe_margin_px', 50).value

        # ------State------
        self.cur_pos = Pose2D()
        self.last_vision_time = None
        self.mast_height = 0.0
        self.target_mast_height = 0.0
        self.mast_target_received = False

        self.state = IDLE
        self.goal = Pose2D()
        self.goal_heading = None
        self.waypoint = (0.0, 0.0)
        self.wheel_dir = 1
        self.final_retries = 0
        self.goal_start_time = 0.0

        self.calib = {
            'drive': {'good': 0, 'bad': 0, 'flips': 0, 'locked': False},
            'steering': {'good': 0, 'bad': 0, 'flips': 0, 'locked': False},
        }
        self.calib_window = None

        # Isaac Sim keeps applying the last Ackermann command, so send zeros for ~1 s at startup.
        self.startup_stop_ticks = 10

        # ------Subscribers------
        self.create_subscription(Pose2D, f'{ns}/vision_data', self.vision_callback, 10)
        self.create_subscription(Pose2D, f'{ns}/lift_target_pos', self.goal_callback, 10)
        self.create_subscription(Twist, f'{ns}/mast_target_pos', self.mast_target_callback, 10)
        self.create_subscription(JointState, f'{ns}/mast_pos', self.mast_callback, 10)
        self.create_subscription(Empty, f'{ns}/stop', self.stop_callback, 10)

        # ------Publishers------
        self.pub_lift = self.create_publisher(Twist, f'{ns}/lift', 10)
        self.pub_drive = self.create_publisher(AckermannDriveStamped, f'{ns}/ackermann', 10)
        self.pub_status = self.create_publisher(String, f'{ns}/goal_status', 10)

        # ------Timers------
        self.create_timer(0.05, self.mast_control_loop)
        self.create_timer(0.1, self.drive_control_loop)

        self.get_logger().info(
            f"Forklift {self.fid} online. Goals: {ns}/lift_target_pos (x, y px, theta deg or NaN), "
            f"lift: {ns}/mast_target_pos, stop: {ns}/stop"
        )

    # ------Helpers------
    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def heading(self) -> float:
        """Direction the forks point (radians, tracker convention)."""
        return wrap(self.cur_pos.theta + self.heading_offset)

    def in_safe_area(self, x: float, y: float) -> bool:
        m = self.safe_margin
        return m <= x <= self.camera_width - m and m <= y <= self.camera_height - m

    def clamp_to_safe_area(self, x: float, y: float):
        m = self.safe_margin
        return clamp(x, m, self.camera_width - m), clamp(y, m, self.camera_height - m)

    def publish_drive(self, speed: float, steering_deg: float):
        cmd = AckermannDriveStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.drive.speed = float(speed)
        cmd.drive.steering_angle = float(steering_deg)
        self.pub_drive.publish(cmd)

    def publish_stop(self):
        self.publish_drive(0.0, 0.0)

    def publish_status(self, text: str):
        self.pub_status.publish(String(data=text))

    # ------Callbacks------
    def vision_callback(self, msg: Pose2D):
        self.cur_pos = msg
        self.last_vision_time = self.now_sec()

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

        if self.goal_heading is None:
            self.waypoint = (self.goal.x, self.goal.y)
        else:
            # Pre-goal point behind the target so the last stretch is driven forks-first along the heading.
            wx = self.goal.x - self.approach_distance * math.cos(self.goal_heading)
            wy = self.goal.y + self.approach_distance * math.sin(self.goal_heading)
            self.waypoint = self.clamp_to_safe_area(wx, wy)

        self.state = APPROACH
        self.final_retries = 0
        self.goal_start_time = self.now_sec()
        self.calib_window = None

        heading_txt = 'any' if self.goal_heading is None else f"{msg.theta:.1f}°"
        self.get_logger().info(
            f"New goal: x={msg.x:.1f}, y={msg.y:.1f}, heading={heading_txt} "
            f"(approach point {self.waypoint[0]:.0f}, {self.waypoint[1]:.0f})"
        )

    def mast_target_callback(self, msg: Twist):
        self.target_mast_height = msg.linear.z
        self.mast_target_received = True
        self.get_logger().info(f"New mast target received: {self.target_mast_height}")

    def mast_callback(self, msg: JointState):
        if msg.position:
            self.mast_height = msg.position[0]

    def stop_callback(self, msg: Empty):
        if self.state != IDLE:
            self.finish(False, 'stopped by user')
        else:
            self.publish_stop()
        self.get_logger().info("Stop command received. Holding position.")

    # ------Lift------
    def mast_control_loop(self):
        if not self.mast_target_received:
            return

        error = self.target_mast_height - self.mast_height
        speed = 0.1  # meters per second

        if abs(error) > 0.05:
            control_signal = self.mast_height + speed * 0.05 + 0.1 * error
            control_signal = clamp(control_signal, -0.15, 2.0)

            lift_cmd = Twist()
            lift_cmd.linear.z = control_signal
            self.pub_lift.publish(lift_cmd)

    # ------Drive------
    def finish(self, success: bool, reason: str):
        self.state = IDLE
        self.publish_stop()
        text = f"{'reached' if success else 'failed'}: {reason}"
        if success:
            self.get_logger().info(f"Goal {text}")
        else:
            self.get_logger().warn(f"Goal {text}")
        self.publish_status(text)

    def drive_control_loop(self):
        if self.startup_stop_ticks > 0:
            self.startup_stop_ticks -= 1
            self.publish_stop()
            return

        if self.state == IDLE:
            return

        now = self.now_sec()
        if self.last_vision_time is None or now - self.last_vision_time > self.vision_timeout:
            self.publish_stop()
            self.calib_window = None
            self.get_logger().warn(
                "No fresh vision data for this forklift; holding until the camera sees it again.",
                throttle_duration_sec=2.0
            )
            return

        if now - self.goal_start_time > self.goal_timeout:
            self.finish(False, f"timed out after {self.goal_timeout:.0f} s")
            return

        x, y = self.cur_pos.x, self.cur_pos.y
        heading = self.heading()
        dist_goal = math.hypot(self.goal.x - x, self.goal.y - y)

        heading_error = None if self.goal_heading is None else wrap(self.goal_heading - heading)
        heading_ok = heading_error is None or abs(heading_error) <= self.heading_tolerance
        detail = f"position error {dist_goal:.1f} px"
        if heading_error is not None:
            detail += f", heading error {math.degrees(heading_error):.1f}°"

        if dist_goal < self.position_tolerance and (self.goal_heading is None or self.state == FINAL) and heading_ok:
            self.finish(True, detail)
            return

        if self.state == APPROACH:
            wx, wy = self.waypoint
            dist_wp = math.hypot(wx - x, wy - y)
            if self.goal_heading is not None and dist_wp < self.approach_tolerance:
                self.state = FINAL
                self.get_logger().info("Approach point reached. Driving in forks-first along the goal heading.")
            else:
                wheel_speed = self.steer_point_toward(-self.tag_offset, wx, wy, dist_wp)

        if self.state == FINAL:
            wheel_speed = self.follow_goal_line()
            if wheel_speed is None:
                if dist_goal <= self.final_tolerance and heading_ok:
                    self.finish(True, detail)
                    return
                if self.final_retries >= self.max_final_retries:
                    self.finish(False, f"{detail} after {self.final_retries} retries")
                    return
                self.final_retries += 1
                self.state = APPROACH
                self.get_logger().info(f"Missed the goal ({detail}), re-approaching (retry {self.final_retries}).")
                return

        direction = 'forks-first' if wheel_speed * self.drive_sign > 0 else 'counterweight-first'
        self.get_logger().info(
            f"[{self.state}] pos=({x:.0f}, {y:.0f}) heading={math.degrees(heading):.0f}° "
            f"dist={dist_goal:.0f}px {direction}",
            throttle_duration_sec=1.0
        )

    def speed_for(self, distance: float) -> float:
        return clamp(self.speed_gain * distance, self.min_speed, self.max_speed)

    def point_on_axis(self, offset: float):
        """Image position of the point `offset` px from the front rollers along the fork direction."""
        h = self.heading()
        k = offset + self.tag_offset
        return self.cur_pos.x + k * math.cos(h), self.cur_pos.y - k * math.sin(h)

    def steer_point_toward(self, offset: float, tx: float, ty: float, distance: float) -> float:
        """Move the body point at `offset` (see point_on_axis) straight toward (tx, ty)."""
        px, py = self.point_on_axis(offset)
        direction = image_angle(tx - px, ty - py)
        speed = self.speed_for(distance)
        return self.drive_point(offset, speed * math.cos(direction), speed * math.sin(direction))

    def follow_goal_line(self):
        """Forks-first along the approach line, steering a point ahead of the front rollers.

        Returns the wheel speed, or None once the forklift has crossed the goal."""
        k = self.lead_offset + self.tag_offset
        gx = self.goal.x + k * math.cos(self.goal_heading)
        gy = self.goal.y - k * math.sin(self.goal_heading)
        sx = self.waypoint[0] + k * math.cos(self.goal_heading)
        sy = self.waypoint[1] - k * math.sin(self.goal_heading)
        seg_len = math.hypot(gx - sx, gy - sy)
        ux, uy = (gx - sx) / seg_len, (gy - sy) / seg_len

        px, py = self.point_on_axis(self.lead_offset)
        along = (px - sx) * ux + (py - sy) * uy
        if along >= seg_len:
            return None

        # Aim past the goal so steering stays steady through the end of the line.
        carrot = along + self.lookahead
        tx, ty = sx + carrot * ux, sy + carrot * uy
        remaining = max(seg_len - along, 0.0)
        return self.steer_point_toward(self.lead_offset, tx, ty, remaining)

    def drive_point(self, offset: float, vx_img: float, vy_img: float) -> float:
        """Command wheel speed/steering so the body point at `offset` px from the front rollers
        (along the forks) moves with velocity (vx_img, vy_img) in the y-up image frame.

        The front rollers do not slip sideways, so v_point = u*x_f + omega*offset*y_f and the
        rear wheel (at -wheelbase) must roll along u*x_f - omega*wheelbase*y_f."""
        h = self.heading()
        vx = math.cos(h) * vx_img + math.sin(h) * vy_img
        vy = -math.sin(h) * vx_img + math.cos(h) * vy_img
        omega = vy / offset
        ax = self.drive_sign * vx
        ay = self.drive_sign * (-omega * self.wheelbase)

        # Pick the wheel direction with some hysteresis to avoid chattering for sideways targets.
        angle = math.atan2(ay, ax)
        if self.wheel_dir > 0 and abs(angle) > math.radians(100):
            self.wheel_dir = -1
        elif self.wheel_dir < 0 and abs(wrap(angle - math.pi)) > math.radians(100):
            self.wheel_dir = 1
        wheel_yaw = angle if self.wheel_dir > 0 else wrap(angle - math.pi)
        wheel_yaw_deg = clamp(math.degrees(wheel_yaw), -self.max_steering, self.max_steering)

        wheel_speed = self.wheel_dir * clamp(math.hypot(ax, ay), self.min_speed, self.max_speed)
        steering = -self.drive_sign * self.steering_sign * wheel_yaw_deg
        self.publish_drive(wheel_speed, steering)

        if self.auto_calibrate:
            self.update_calibration(wheel_speed, steering)
        return wheel_speed

    # ------Sign self-check------
    def update_calibration(self, wheel_speed: float, steering: float):
        """Compare observed motion with the sign model and flip a sign if it is clearly wrong."""
        x, y, h = self.cur_pos.x, self.cur_pos.y, self.cur_pos.theta
        steer_dir = 0 if abs(steering) < 15.0 else (1 if steering > 0 else -1)
        key = (wheel_speed > 0, steer_dir)

        w = self.calib_window
        if w is None or w['key'] != key:
            self.calib_window = {'key': key, 'x': x, 'y': y, 'h': h}
            return

        dx, dy = x - w['x'], y - w['y']
        if math.hypot(dx, dy) < 30.0:
            return
        self.calib_window = None

        dh = wrap(h - w['h'])
        wheel_dir = 1 if wheel_speed > 0 else -1

        if steer_dir == 0:
            mid_heading = wrap(w['h'] + dh / 2 + self.heading_offset)
            forks_first = wheel_speed * self.drive_sign > 0
            expected = mid_heading if forks_first else mid_heading + math.pi
            c = math.cos(image_angle(dx, dy) - expected)
            if c < -0.5:
                self.record_calibration('drive', False)
            elif c > 0.5:
                self.record_calibration('drive', True)
        elif abs(dh) > math.radians(8):
            expected_sign = self.steering_sign * wheel_dir * steer_dir
            self.record_calibration('steering', (dh > 0) == (expected_sign > 0))

    def record_calibration(self, which: str, consistent: bool):
        c = self.calib[which]
        if c['locked']:
            return
        if consistent:
            c['good'] += 1
            c['bad'] = 0
            if c['good'] >= 2:
                c['locked'] = True
                self.get_logger().info(f"{which}_sign confirmed by observed motion.")
            return

        c['bad'] += 1
        c['good'] = 0
        if c['bad'] < 2:
            return
        c['bad'] = 0
        if c['flips'] >= 2:
            self.get_logger().warn(f"{which}_sign keeps disagreeing with the camera; leaving it unchanged.")
            c['locked'] = True
            return
        c['flips'] += 1
        if which == 'drive':
            self.drive_sign = -self.drive_sign
            value = self.drive_sign
        else:
            self.steering_sign = -self.steering_sign
            value = self.steering_sign
        self.get_logger().warn(
            f"Observed motion contradicts {which}_sign; switched it to {value:+.0f}. "
            f"Launch with '--ros-args -p {which}_sign:={value:.1f}' to make this permanent."
        )


def main(forklift_id: int, args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = ForkliftController(forklift_id)
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        if node is not None:
            if rclpy.ok():
                node.publish_stop()
            node.get_logger().info(f"Shutting down Forklift {forklift_id}...")
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
