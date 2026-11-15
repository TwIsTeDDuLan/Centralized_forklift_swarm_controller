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
        # Max tag motion / turn over the settle window that still counts as stopped.
        self.settle_motion_px = p('settle_motion_px', 5.0).value
        self.settle_turn_deg = p('settle_turn_deg', 3.5).value
        self._settle_from = None
        self.auto_calibrate = False
        self.heading_offset = math.radians(p('tag_heading_offset_deg', 0.0).value)
        self.auto_heading_check = p('auto_heading_check', True).value
        self.heading_check = {'window': None, 'votes': [], 'locked': False}
        self.auto_insert = p('auto_insert', True).value
        # Insert is done once the forks moved (tip-to-face distance + insert_depth_m) from where it began.
        # The face distance is read from the scan's level channel when the insert starts.
        self.insert_depth_m = p('insert_depth_m', 0.65).value  # training: success 0.40, target 0.70, back wall 0.76
        self._insert_goal = None
        self.insert_timeout = p('insert_timeout_sec', 20.0).value

        self.nav_policy = NumpyPolicy(nav_path)
        nav_meta = self.nav_policy.meta
        self.max_speed = nav_meta['max_wheel_speed_deg']
        self.max_steering = nav_meta['max_steer_deg']
        # General navigation tolerances:
        self.position_tolerance = p('rl_position_tolerance_px', max(25.0, nav_meta['pos_tolerance'] * 1.5 * self.px_per_m)).value
        self.heading_tolerance = math.radians(p('rl_heading_tolerance_deg', 20.0).value)

        # Staging tolerances for auto_insert (decomposed into pocket-aligned frame):
        # Longitudinal: along fork axis (generous: 24 px ~ 0.37 m).
        # Lateral: perpendicular to fork axis (strict: 6.5 px ~ 0.10 m, ensuring tines enter 0.40 m pockets).
        # Heading: parallel to pocket axis (strict: 7.0°).
        self.staging_longitudinal_tol = p('rl_staging_long_tol_px', 24.0).value
        self.staging_lateral_tol = p('rl_staging_lat_tol_px', 6.5).value
        self.staging_heading_tol = math.radians(p('rl_staging_heading_tol_deg', 7.0).value)
        # Fallback Euclidean staging tolerance if no heading constraint:
        self.staging_position_tolerance = p('rl_staging_pos_tolerance_px', 26.0).value

        # Aerial approach corridor parameters for distant spawns:
        self.corridor_approach_dist_m = p('corridor_approach_dist_m', 4.0).value  # standoff before staging (4.0 m)
        self.corridor_lookahead_m = p('corridor_lookahead_m', 1.2).value          # lookahead along centerline (1.2 m)
        self._staging_phase = 'corridor'  # 'approach', 'corridor', or 'realign'

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
        self.place_timeout = p('place_timeout_sec', 25.0).value
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
        self._settle_from = None

        is_staging = (not self.carrying and self.auto_insert and self.insert_policy is not None and self.goal_heading is not None)
        if is_staging:
            v_hat = (math.cos(self.goal_heading), -math.sin(self.goal_heading))
            n_hat = (-math.sin(self.goal_heading), -math.cos(self.goal_heading))
            dx = self.cur_pos.x - self.goal.x
            dy = self.cur_pos.y - self.goal.y
            d_par = -(dx * v_hat[0] + dy * v_hat[1])
            e_lat = dx * n_hat[0] + dy * n_hat[1]
            e_yaw = abs(wrap(self.goal_heading - self.heading()))
            in_corridor = (0.0 <= d_par <= 2.5 * self.px_per_m and
                           abs(e_lat) <= 0.35 * self.px_per_m and
                           e_yaw <= math.radians(25.0))
            self._staging_phase = 'corridor' if in_corridor else 'approach'
            if self._staging_phase == 'approach':
                self.get_logger().info(
                    f"Distant spawn (dist={math.hypot(dx, dy)/self.px_per_m:.2f} m, "
                    f"lateral={e_lat/self.px_per_m:.2f} m, heading error={math.degrees(e_yaw):.1f}°). "
                    f"Engaging aerial corridor approach."
                )
        else:
            self._staging_phase = 'corridor'

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

        is_staging = (not self.carrying and self.auto_insert and self.insert_policy is not None and self.goal_heading is not None)

        if is_staging:
            # Vector along forks into the pockets:
            v_hat = (math.cos(self.goal_heading), -math.sin(self.goal_heading))
            n_hat = (-math.sin(self.goal_heading), -math.cos(self.goal_heading))
            dx = self.cur_pos.x - self.goal.x
            dy = self.cur_pos.y - self.goal.y
            d_par = -(dx * v_hat[0] + dy * v_hat[1])  # px along centerline ahead of staging
            e_lat = dx * n_hat[0] + dy * n_hat[1]     # px lateral offset from centerline
            heading_error = wrap(self.goal_heading - self.heading())

            detail = (f"longitudinal {d_par/self.px_per_m:+.2f} m, "
                      f"lateral {e_lat/self.px_per_m:+.2f} m, "
                      f"heading error {math.degrees(heading_error):.1f}°")

            # Check if ready to transition from 'approach' to 'corridor'
            if self._staging_phase == 'approach':
                app_dist_px = self.corridor_approach_dist_m * self.px_per_m
                app_x = clamp(self.goal.x - app_dist_px * v_hat[0], self.safe_margin, self.camera_width - self.safe_margin)
                app_y = clamp(self.goal.y - app_dist_px * v_hat[1], self.safe_margin, self.camera_height - self.safe_margin)
                dist_to_app = math.hypot(self.cur_pos.x - app_x, self.cur_pos.y - app_y)

                at_app_waypoint = (dist_to_app < 35.0 and abs(heading_error) < math.radians(18.0))
                already_in_corridor = (0.5 * self.px_per_m <= d_par <= (self.corridor_approach_dist_m + 0.5) * self.px_per_m and
                                       abs(e_lat) <= 0.20 * self.px_per_m and
                                       abs(heading_error) < math.radians(15.0))
                if at_app_waypoint or already_in_corridor:
                    self._staging_phase = 'corridor'
                    self.get_logger().info(
                        f"Aerial corridor entered ({detail}). Switching to pocket centerline tracking."
                    )

            if self._staging_phase == 'approach':
                app_dist_px = self.corridor_approach_dist_m * self.px_per_m
                target_x = clamp(self.goal.x - app_dist_px * v_hat[0], self.safe_margin, self.camera_width - self.safe_margin)
                target_y = clamp(self.goal.y - app_dist_px * v_hat[1], self.safe_margin, self.camera_height - self.safe_margin)
                target_heading = self.goal_heading
            elif self._staging_phase == 'realign':
                realign_dist_px = 2.0 * self.px_per_m
                target_x = clamp(self.goal.x - realign_dist_px * v_hat[0], self.safe_margin, self.camera_width - self.safe_margin)
                target_y = clamp(self.goal.y - realign_dist_px * v_hat[1], self.safe_margin, self.camera_height - self.safe_margin)
                target_heading = self.goal_heading
                if d_par >= 1.7 * self.px_per_m and abs(heading_error) < math.radians(15.0):
                    self._staging_phase = 'corridor'
                    self.get_logger().info(f"Re-aligned along corridor ({detail}). Resuming approach.")
            else:
                # Corridor phase: Active Aerial Pocket Alignment Servoing
                # Uses the overhead camera feed to servo the forklift straight down the pocket entry centerline.
                long_ok = abs(d_par) <= self.staging_longitudinal_tol
                lat_ok = abs(e_lat) <= self.staging_lateral_tol
                heading_ok = abs(heading_error) <= self.staging_heading_tol

                if long_ok and lat_ok and heading_ok:
                    self.publish_stop()
                    if self.inside_ticks == 0:
                        self._settle_from = (self.cur_pos.x, self.cur_pos.y, self.heading())
                    self.inside_ticks += 1
                    sx, sy, sh = self._settle_from
                    moving = (math.hypot(self.cur_pos.x - sx, self.cur_pos.y - sy) > self.settle_motion_px
                              or abs(wrap(self.heading() - sh)) > math.radians(self.settle_turn_deg))
                    if moving:
                        self._settle_from = (self.cur_pos.x, self.cur_pos.y, self.heading())
                        self.inside_ticks = 0
                    elif self.inside_ticks >= self.settle_ticks:
                        self.state = INSERT
                        self._policy_owns_lift = True
                        self._insert_start = now
                        self._staging_start = (self.cur_pos.x, self.cur_pos.y)
                        self._insert_goal = None
                        self.last_insert_action[:] = 0.0
                        self.publish_status("reached: staging")
                        self.get_logger().info(
                            f"Staging reached with aerial pocket alignment ({detail}); switching to insert policy."
                        )
                        return
                    return
                else:
                    self.inside_ticks = 0
                    self._settle_from = None

                # Arrived at staging standoff but misaligned: back out to re-enter corridor cleanly
                if d_par <= 0.20 * self.px_per_m and (abs(e_lat) > self.staging_lateral_tol or abs(heading_error) > self.staging_heading_tol):
                    self._staging_phase = 'realign'
                    self.get_logger().warn(
                        f"Staging pose reached with lateral/heading offset ({detail}); backing out to re-align."
                    )
                    return

                # Front roller axle position (fixed non-swiveling wheels, immune to rear tail-swing):
                front_x = self.cur_pos.x + self.tag_offset * math.cos(self.heading())
                front_y = self.cur_pos.y - self.tag_offset * math.sin(self.heading())
                front_goal_x = self.goal.x + self.tag_offset * math.cos(self.goal_heading)
                front_goal_y = self.goal.y - self.tag_offset * math.sin(self.goal_heading)
                dfx = front_x - front_goal_x
                dfy = front_y - front_goal_y
                e_lat_front = dfx * n_hat[0] + dfy * n_hat[1]
                e_lat_front_m = e_lat_front / self.px_per_m

                # Active aerial line tracking down the pocket axis (Stanley tracking at front axle):
                e_yaw_rad = heading_error
                align_factor = clamp(1.0 - abs(e_lat_front_m) * 2.0 - abs(e_yaw_rad) * 1.0, 0.3, 1.0)
                spd_mag = clamp(abs(d_par) * 2.5 * align_factor, self.min_speed, self.max_speed)
                speed = -spd_mag if d_par > 0 else spd_mag * 0.5
                v_ms = max(0.15, abs(speed) / self.max_speed * 0.67)
                theta_intercept = math.atan2(-1.2 * e_lat_front_m, v_ms)
                steer_cmd = clamp(math.degrees(2.0 * e_yaw_rad + theta_intercept), -self.max_steering, self.max_steering)

                self.last_action = np.array([speed / self.max_speed, steer_cmd / self.max_steering])
                self.publish_drive(speed, steer_cmd)
                self.get_logger().info(
                    f"[aerial-corridor] pos=({self.cur_pos.x:.0f}, {self.cur_pos.y:.0f}) heading={math.degrees(self.heading()):.0f}° "
                    f"{detail} cmd speed={speed:.0f} steer={steer_cmd:.0f}",
                    throttle_duration_sec=1.0
                )
                return

        else:
            # Standard navigation (delivery goal or non-pallet waypoint)
            dist_px = math.hypot(self.goal.x - self.cur_pos.x, self.goal.y - self.cur_pos.y)
            heading_error = None if self.goal_heading is None else wrap(self.goal_heading - self.heading())
            heading_ok = heading_error is None or abs(heading_error) <= self.heading_tolerance
            detail = f"position error {dist_px:.1f} px"
            if heading_error is not None:
                detail += f", heading error {math.degrees(heading_error):.1f}°"

            if dist_px < self.position_tolerance and heading_ok:
                if self.inside_ticks == 0:
                    self._settle_from = (self.cur_pos.x, self.cur_pos.y, self.heading())
                self.inside_ticks += 1
                sx, sy, sh = self._settle_from
                moving = (math.hypot(self.cur_pos.x - sx, self.cur_pos.y - sy) > self.settle_motion_px
                          or abs(wrap(self.heading() - sh)) > math.radians(self.settle_turn_deg))
                if moving:
                    self._settle_from = (self.cur_pos.x, self.cur_pos.y, self.heading())
                    self.inside_ticks = 0
                elif self.inside_ticks >= self.settle_ticks:
                    if self.carrying:
                        self.finish(True, f"delivered ({detail})")
                        return
                    self.finish(True, f"staging ({detail})")
                    return
            else:
                self.inside_ticks = 0
                self._settle_from = None

            target_x = self.goal.x
            target_y = self.goal.y
            target_heading = self.goal_heading

        goal_yaw = None if target_heading is None else wrap(target_heading - math.pi)
        obs = build_actor_obs(self.nav_policy.meta, self.to_meters(self.cur_pos.x, self.cur_pos.y), self.body_yaw(),
                              self.to_meters(target_x, target_y), goal_yaw, self.last_action)
        action = self.nav_policy(obs)
        self.last_action = action
        speed = clamp(action[0] * self.max_speed, -self.max_speed, self.max_speed)
        steering = clamp(action[1] * self.max_steering, -self.max_steering, self.max_steering)
        self.publish_drive(speed, steering)
        if self.auto_heading_check:
            self.check_heading_offset(speed, steering)
        mode_str = f"nav-{self._staging_phase}" if is_staging else "nav"
        self.get_logger().info(
            f"[{mode_str}] pos=({self.cur_pos.x:.0f}, {self.cur_pos.y:.0f}) heading={math.degrees(self.heading()):.0f}° "
            f"{detail} cmd speed={speed:.0f} steer={steering:.0f}",
            throttle_duration_sec=1.0
        )

    def _insert_tick(self, now: float):
        elapsed = now - (self._insert_start or now)
        if elapsed > self.insert_timeout:
            self.finish(False, f"insert timed out after {self.insert_timeout:.0f} s")
            return
        if self._scan is None or self._scan_time is None or now - self._scan_time > 0.5:
            self.publish_stop()
            self.get_logger().warn("No fresh /scan for insert policy; holding.", throttle_duration_sec=2.0)
            return

        if self._insert_goal is None:
            self._insert_goal = self.insert_travel(self._scan)
            if self._insert_goal is None:
                if elapsed > 3.0:
                    self.finish(False, "pallet face not in the fork scan at staging; not inserting blind")
                    return
        if self._staging_start is not None and self._insert_goal is not None:
            dx = (self.cur_pos.x - self._staging_start[0]) / self.px_per_m
            dy = -(self.cur_pos.y - self._staging_start[1]) / self.px_per_m
            progress = dx * math.cos(self.heading()) + dy * math.sin(self.heading())
            if progress >= self._insert_goal:
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

    def insert_travel(self, ranges):
        """Distance to drive along the forks: fork tips to the pallet face (from the scan) + insert_depth_m."""
        meta = self.insert_policy.meta
        n_az = 64
        lidar_behind_tips = 0.138  # LIDAR_OFFSET_L is 13.8 cm behind the tips along the forks
        r = np.asarray(ranges, dtype=np.float64).reshape(-1, n_az)
        max_r = meta['lidar_max_range']
        hit = r < 0.95 * max_r
        if not hit.any():
            return None
        az = np.linspace(-math.radians(45.0), math.radians(45.0), n_az)
        forward_dists = []
        for ch in range(r.shape[0]):
            ch_hit = hit[ch]
            if ch_hit.any():
                forward_dists.append((r[ch] * np.cos(az))[ch_hit].min())
        if not forward_dists:
            return None
        face = float(min(forward_dists))
        travel = max(0.0, face - lidar_behind_tips) + self.insert_depth_m
        self.get_logger().info(f"Pallet face {face - lidar_behind_tips:.2f} m ahead of the tips; "
                               f"inserting until the forks moved {travel:.2f} m.")
        return travel

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
        self.publish_drive(self.max_speed, 0.0)

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
