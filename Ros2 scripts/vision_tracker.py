#!/usr/bin/env python3

import os
import sys

# Add ROS Python packages path if the shell has not sourced ROS setup.
ros_python_path = '/opt/ros/jazzy/lib/python3.12/site-packages'
if os.path.isdir(ros_python_path) and ros_python_path not in sys.path:
    sys.path.append(ros_python_path)

import rclpy
import cv2
import cv2.aruco as aruco
import numpy as np
import math
import time

from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose2D


class SwarmTracker(Node):
    def __init__(self):
        super().__init__('swarm_vision_tracker')
        
        # ------Filter & Constant-Velocity Predictor Parameters------
        self.heading_filter_alpha = 0.35  # Heading low-pass weight on new measurement
        self.pos_filter_alpha = 0.50      # Position low-pass weight on new measurement
        self.pallet_alpha = 0.15          # Stronger smoothing for static pallets
        self.max_coast_sec = 1.0          # Coast up to 1.0 s of missed marker frames
        self.max_jump_px = 32.0           # Discard single-frame outliers (> 0.5 m at 65 px/m)
        self.max_speed_px_s = 80.0        # Physical speed clamp (~1.2 m/s)
        self.max_ang_speed_rad_s = 3.5    # Physical angular speed clamp

        # Per-tag state dictionary:
        # tag_id -> {
        #   'x': float, 'y': float, 'heading': float,
        #   'vx': float, 'vy': float, 'vtheta': float,
        #   'last_seen_time': float, 'last_update_time': float,
        #   'is_static': bool
        # }
        self.tracks = {}
        self.last_frame_time = None
        # ------------------------------------------------------------
        
        # ------Subscribers------
        self.subscription = self.create_subscription(
            Image,
            '/warehouse/overhead_cam/image_raw',
            self.image_callback,
            10)
        
        # ------Publishers------
        # For Forklifts
        self.pub_fl1_vision = self.create_publisher(Pose2D, '/forklift_1/vision_data', 10)
        self.pub_fl2_vision = self.create_publisher(Pose2D, '/forklift_2/vision_data', 10)
        self.pub_fl3_vision = self.create_publisher(Pose2D, '/forklift_3/vision_data', 10)

        # For Pallets
        self.pub_plt1_vision = self.create_publisher(Pose2D, '/pallet_1/vision_data', 10)
        self.pub_plt2_vision = self.create_publisher(Pose2D, '/pallet_2/vision_data', 10)
        self.pub_plt3_vision = self.create_publisher(Pose2D, '/pallet_3/vision_data', 10)

        self.bridge = CvBridge()
        self.disable_aruco = os.getenv('SWARM_DISABLE_ARUCO', '0').lower() in ('1', 'true', 'yes', 'on')
        
        # ArUco dictionary (DICT_4X4_50)
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        if hasattr(aruco, 'DetectorParameters'):
            self.aruco_params = aruco.DetectorParameters()
        else:
            self.aruco_params = aruco.DetectorParameters_create()
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params) if hasattr(aruco, 'ArucoDetector') else None

        if self.disable_aruco:
            self.get_logger().warn("Swarm Tracker is online with ArUco detection disabled (SWARM_DISABLE_ARUCO=1).")
        else:
            self.get_logger().info("Swarm Tracker online with CV Dead-Reckoning & Low-Pass Filtering (Fix B active).")

    def image_callback(self, msg):
        now = self.get_clock().now().nanoseconds / 1e9
        dt_frame = 0.033 if self.last_frame_time is None else min(0.1, max(0.001, now - self.last_frame_time))
        self.last_frame_time = now

        # Convert ROS Image to OpenCV Image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        if cv_image is None:
            return

        if len(cv_image.shape) == 2:
            gray = cv_image
            draw_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
        elif len(cv_image.shape) == 3:
            channels = cv_image.shape[2]
            if channels == 4:
                draw_image = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2BGR)
            elif channels == 3:
                draw_image = cv_image
            else:
                self.get_logger().warn(f"Unsupported image channels: {channels}")
                return
            gray = cv2.cvtColor(draw_image, cv2.COLOR_BGR2GRAY)
        else:
            self.get_logger().warn(f"Unsupported image shape: {cv_image.shape}")
            return

        if gray.dtype != np.uint8:
            gray = cv2.convertScaleAbs(gray)

        gray = np.ascontiguousarray(gray)
        draw_image = np.ascontiguousarray(draw_image)

        if gray.size == 0 or self.disable_aruco:
            cv2.imshow("Swarm God-Camera", draw_image)
            cv2.waitKey(1)
            return

        # Detect ArUco tags
        if self.detector is not None:
            corners, ids, rejected = self.detector.detectMarkers(gray)
        else:
            corners, ids, rejected = aruco.detectMarkers(gray, self.aruco_dict)

        detected_ids = set()

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(draw_image, corners, ids)

            for i in range(len(ids)):
                tag_id = int(ids[i][0])
                detected_ids.add(tag_id)
                tag_corners = corners[i][0]

                c0 = tag_corners[0]  # Top-Left
                c1 = tag_corners[1]  # Top-Right
                c2 = tag_corners[2]  # Bottom-Right
                c3 = tag_corners[3]  # Bottom-Left

                # Exact float center
                center_x_float = float((c0[0] + c2[0]) / 2.0)
                center_y_float = float((c0[1] + c2[1]) / 2.0)

                # Midpoint of front edge
                front_x_float = float((c0[0] + c1[0]) / 2.0)
                front_y_float = float((c0[1] + c1[1]) / 2.0)

                # Heading in radians (OpenCV Y down -> ROS Y up)
                dx = front_x_float - center_x_float
                dy = center_y_float - front_y_float
                raw_heading = math.atan2(dy, dx)

                is_static = (tag_id > 3)  # Pallets 4..6 are static

                # Track initialization or update
                if tag_id not in self.tracks:
                    self.tracks[tag_id] = {
                        'x': center_x_float,
                        'y': center_y_float,
                        'heading': raw_heading,
                        'vx': 0.0,
                        'vy': 0.0,
                        'vtheta': 0.0,
                        'last_seen_time': now,
                        'last_update_time': now,
                        'is_static': is_static
                    }
                else:
                    tr = self.tracks[tag_id]
                    dt = max(0.001, now - tr['last_seen_time'])

                    if is_static:
                        # Pallet: strongly filter, zero velocity
                        tr['x'] = (1.0 - self.pallet_alpha) * tr['x'] + self.pallet_alpha * center_x_float
                        tr['y'] = (1.0 - self.pallet_alpha) * tr['y'] + self.pallet_alpha * center_y_float
                        dh = math.atan2(math.sin(raw_heading - tr['heading']), math.cos(raw_heading - tr['heading']))
                        tr['heading'] = math.atan2(
                            math.sin(tr['heading'] + self.pallet_alpha * dh),
                            math.cos(tr['heading'] + self.pallet_alpha * dh)
                        )
                    else:
                        # Forklift: Constant-Velocity prediction + Outlier Rejection + Smoothing
                        pred_x = tr['x'] + tr['vx'] * dt
                        pred_y = tr['y'] + tr['vy'] * dt
                        jump = math.hypot(center_x_float - pred_x, center_y_float - pred_y)

                        # Outlier rejection: dampen impossible single-frame jumps (> 32 px ~ 0.5 m)
                        if jump > self.max_jump_px and (now - tr['last_seen_time'] < 0.3):
                            meas_x = pred_x + 0.15 * (center_x_float - pred_x)
                            meas_y = pred_y + 0.15 * (center_y_float - pred_y)
                        else:
                            meas_x = center_x_float
                            meas_y = center_y_float

                        new_x = (1.0 - self.pos_filter_alpha) * pred_x + self.pos_filter_alpha * meas_x
                        new_y = (1.0 - self.pos_filter_alpha) * pred_y + self.pos_filter_alpha * meas_y

                        # Velocity estimation (EMA)
                        if dt < 0.25:
                            inst_vx = (new_x - tr['x']) / dt
                            inst_vy = (new_y - tr['y']) / dt
                            spd = math.hypot(inst_vx, inst_vy)
                            if spd > self.max_speed_px_s:
                                inst_vx = (inst_vx / spd) * self.max_speed_px_s
                                inst_vy = (inst_vy / spd) * self.max_speed_px_s
                            tr['vx'] = 0.65 * tr['vx'] + 0.35 * inst_vx
                            tr['vy'] = 0.65 * tr['vy'] + 0.35 * inst_vy

                        tr['x'] = new_x
                        tr['y'] = new_y

                        # Heading smoothing with circular difference
                        dh = math.atan2(math.sin(raw_heading - tr['heading']), math.cos(raw_heading - tr['heading']))
                        if dt < 0.25:
                            inst_w = dh / dt
                            inst_w = max(-self.max_ang_speed_rad_s, min(self.max_ang_speed_rad_s, inst_w))
                            tr['vtheta'] = 0.65 * tr['vtheta'] + 0.35 * inst_w

                        tr['heading'] = math.atan2(
                            math.sin(tr['heading'] + self.heading_filter_alpha * dh),
                            math.cos(tr['heading'] + self.heading_filter_alpha * dh)
                        )

                    tr['last_seen_time'] = now
                    tr['last_update_time'] = now

                # Publish filtered pose
                tr = self.tracks[tag_id]
                self.publish_tag(tag_id, tr['x'], tr['y'], tr['heading'])

                # Render center dot and heading vector
                cx, cy = int(tr['x']), int(tr['y'])
                cv2.circle(draw_image, (cx, cy), 5, (0, 0, 255), -1)
                hx = int(cx + 25 * math.cos(tr['heading']))
                hy = int(cy - 25 * math.sin(tr['heading']))
                cv2.line(draw_image, (cx, cy), (hx, hy), (0, 255, 0), 2)
                h_deg = math.degrees(tr['heading'])
                cv2.putText(draw_image, f"id={tag_id} {h_deg:.0f}deg", (cx - 20, cy - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        # Coast undetected tags (Dead-Reckoning up to max_coast_sec)
        for tag_id, tr in list(self.tracks.items()):
            if tag_id not in detected_ids:
                time_since_seen = now - tr['last_seen_time']
                if time_since_seen <= self.max_coast_sec:
                    dt = max(0.001, now - tr['last_update_time'])
                    tr['last_update_time'] = now

                    if not tr['is_static']:
                        # Coast forklift forward using estimated velocity
                        tr['x'] += tr['vx'] * dt
                        tr['y'] += tr['vy'] * dt
                        tr['heading'] = math.atan2(
                            math.sin(tr['heading'] + tr['vtheta'] * dt),
                            math.cos(tr['heading'] + tr['vtheta'] * dt)
                        )
                        # Gently bleed velocity during coasting
                        tr['vx'] *= 0.95
                        tr['vy'] *= 0.95
                        tr['vtheta'] *= 0.95

                    # Publish continuous coasted pose so downstream controllers never stall
                    self.publish_tag(tag_id, tr['x'], tr['y'], tr['heading'])

                    # Visual feedback: Orange circle indicating active coasting
                    cx, cy = int(tr['x']), int(tr['y'])
                    cv2.circle(draw_image, (cx, cy), 8, (0, 165, 255), 2)
                    cv2.putText(draw_image, f"id={tag_id} [COAST {time_since_seen:.2f}s]", (cx - 30, cy - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
                else:
                    # After max_coast_sec, stop predicting to avoid phantom tracking
                    pass

        # Show the live tracking window
        cv2.imshow("Swarm God-Camera", draw_image)
        cv2.waitKey(1)

    def publish_tag(self, tag_id: int, x: float, y: float, heading: float):
        if tag_id <= 3:
            self.forklift_vission_data_publisher(tag_id, x, y, heading)
        elif tag_id > 3:
            self.pallet_vission_data_publisher(tag_id, x, y, heading)

    def forklift_vission_data_publisher(self, tag_id: int, x: float, y: float, heading: float):
        msg = Pose2D(x=float(x), y=float(y), theta=float(heading))
        if tag_id == 1:
            self.pub_fl1_vision.publish(msg)
        elif tag_id == 2:
            self.pub_fl2_vision.publish(msg)
        elif tag_id == 3:
            self.pub_fl3_vision.publish(msg)
        else:
            self.get_logger().warn(f"Unknown forklift tag ID: {tag_id}")

    def pallet_vission_data_publisher(self, tag_id: int, x: float, y: float, heading: float):
        msg = Pose2D(x=float(x), y=float(y), theta=float(heading))
        if tag_id == 4:
            self.pub_plt1_vision.publish(msg)
        elif tag_id == 5:
            self.pub_plt2_vision.publish(msg)
        elif tag_id == 6:
            self.pub_plt3_vision.publish(msg)
        else:
            self.get_logger().warn(f"Unknown pallet tag ID: {tag_id}")


def main(args=None):
    rclpy.init(args=args)
    tracker = SwarmTracker()
    rclpy.spin(tracker)
    tracker.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
