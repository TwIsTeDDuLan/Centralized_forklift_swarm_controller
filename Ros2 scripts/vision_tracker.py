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


from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose2D


class SwarmTracker(Node):
    def __init__(self):
        super().__init__('swarm_vision_tracker')
        
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
        self.pub_plt1_vision = self.create_publisher(Pose2D, 'pallet_1/vision_data', 10)
        self.pub_plt2_vision = self.create_publisher(Pose2D, 'pallet_2/vision_data', 10)
        self.pub_plt3_vision = self.create_publisher(Pose2D, 'pallet_3/vision_data', 10)

        # ----------------------

        self.bridge = CvBridge()
        self.disable_aruco = os.getenv('SWARM_DISABLE_ARUCO', '0').lower() in ('1', 'true', 'yes', 'on')
        
        # 2. Set up the ArUco dictionary (Must match the tags you generated!)
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        # OpenCV < 4.7 uses DetectorParameters_create(), while newer builds expose DetectorParameters().
        if hasattr(aruco, 'DetectorParameters'):
            self.aruco_params = aruco.DetectorParameters()
        else:
            self.aruco_params = aruco.DetectorParameters_create()
        # OpenCV 4.7+ provides ArucoDetector; older ROS OpenCV builds use detectMarkers().
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params) if hasattr(aruco, 'ArucoDetector') else None

        if self.disable_aruco:
            self.get_logger().warn("Swarm Tracker is online with ArUco detection disabled (SWARM_DISABLE_ARUCO=1).")
        else:
            self.get_logger().info("Swarm Tracker is online. Scanning for forklifts...")

    def image_callback(self, msg):
        # Convert ROS Image to OpenCV Image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

        # Normalize to a contiguous uint8 grayscale image before ArUco detection.
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
        self.get_logger().debug(f"Received image with shape: {cv_image.shape}, dtype: {cv_image.dtype}")

        if gray.size == 0:
            return
        
        if not self.disable_aruco:
            # Search for tags
            if self.detector is not None:
                corners, ids, rejected = self.detector.detectMarkers(gray)
            else:
                corners, ids, rejected = aruco.detectMarkers(gray, self.aruco_dict)

            if ids is not None:
                # Draw boxes around the detected forklifts
                cv2.aruco.drawDetectedMarkers(draw_image, corners, ids)
                
                # Loop through every tag found
                for i in range(len(ids)):
                    tag_id = ids[i][0]
                    tag_corners = corners[i][0]
                    
                    # 1. Extract the four corners
                    c0 = tag_corners[0] # Top-Left
                    c1 = tag_corners[1] # Top-Right
                    c2 = tag_corners[2] # Bottom-Right
                    c3 = tag_corners[3] # Bottom-Left
                    
                    # 2. Calculate the exact center point (X, Y)
                    center_x = int((c0[0] + c2[0]) / 2)
                    center_y = int((c0[1] + c2[1]) / 2)
                    
                    # 3. Calculate the midpoint of the "Front" edge (between c0 and c1)
                    front_x = int((c0[0] + c1[0]) / 2)
                    front_y = int((c0[1] + c1[1]) / 2)
                    
                    # 4. Calculate the Heading Angle using atan2
                    # Note: We invert the Y-axis because OpenCV Y goes DOWN, but ROS math expects Y to go UP
                    dx = front_x - center_x
                    dy = center_y - front_y 
                    heading_rad = math.atan2(dy, dx)
                    heading_deg = math.degrees(heading_rad)
                    print(f"Heading (rad): {heading_rad:.2f}, Heading (deg): {heading_deg:.1f}")
                    
                    if tag_id <= 3:
                        # publish the vision data for this forklift
                        self.forklift_vission_data_publisher(tag_id, center_x, center_y, heading_rad)

                        # Print the full Pose to the terminal
                        # print(f"Forklift [{tag_id}] -> X: {center_x}, Y: {center_y}, Heading: {heading_deg:.1f}°")
        
                    elif tag_id > 3:
                        # publish the vision data for this Pallet
                        self.pallet_vission_data_publisher(tag_id, center_x, center_y, heading_rad)

                        # Print the full Pose to the terminal
                        # print(f"Pallet [{tag_id}] -> X: {center_x}, Y: {center_y}, Heading: {heading_deg:.1f}°")

                    # Draw a red dot in the center
                    cv2.circle(draw_image, (center_x, center_y), 5, (0, 0, 255), -1)
                                    
                    # Draw a bright green line pointing to the front (The "Heading Vector")
                    cv2.line(draw_image, (center_x, center_y), (front_x, front_y), (0, 255, 0), 3)


        # Show the live tracking window
        cv2.imshow("Swarm God-Camera", draw_image)
        cv2.waitKey(1)

    def forklift_vission_data_publisher(self, tag_id, x, y, heading):
        if tag_id == 1:
            msg = Pose2D(x=float(x), y=float(y), theta=float(heading))
            self.pub_fl1_vision.publish(msg)
        elif tag_id == 2:
            msg = Pose2D(x=float(x), y=float(y), theta=float(heading))
            self.pub_fl2_vision.publish(msg)
        elif tag_id == 3:
            msg = Pose2D(x=float(x), y=float(y), theta=float(heading))
            self.pub_fl3_vision.publish(msg)
        else:
            self.get_logger().warn(f"Unknown tag ID: {tag_id}. Cannot publish vision data.")    

    def pallet_vission_data_publisher(self, tag_id, x, y, heading):
        if tag_id == 4:
            msg = Pose2D(x=float(x), y=float(y), theta=float(heading))
            self.pub_plt1_vision.publish(msg)
        elif tag_id == 5:
            msg = Pose2D(x=float(x), y=float(y), theta=float(heading))
            self.pub_plt3_vision.publish(msg)
        elif tag_id == 6:
            msg = Pose2D(x=float(x), y=float(y), theta=float(heading))
            self.pub_plt3_vision.publish(msg)
        else:
            self.get_logger().warn(f"Unknown tag ID: {tag_id}. Cannot publish vision data.")    


def main(args=None):
    rclpy.init(args=args)
    tracker = SwarmTracker()
    rclpy.spin(tracker)
    tracker.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()