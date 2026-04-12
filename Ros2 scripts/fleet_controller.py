#!/usr/bin/env python3

import os
import sys

# Add ROS Python packages path if the shell has not sourced ROS setup.
ros_python_path = '/opt/ros/jazzy/lib/python3.12/site-packages'
if os.path.isdir(ros_python_path) and ros_python_path not in sys.path:
    sys.path.append(ros_python_path)

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist, Pose2D
from ackermann_msgs.msg import AckermannDriveStamped

import numpy as np
import math

class FleetController(Node):
    def __init__(self):
        super().__init__('fleet_controller')

        # ------Subscribers------
        self.forklift_subscription = self.create_subscription(
            Pose2D,
            '/forklift_1/vision_data',
            self.forklift_1_vision_callback,
            10
        ) 

def main(args=None):
    rclpy.init(args=args)
    fleet_controller = FleetController()
    rclpy.spin(fleet_controller)
    fleet_controller.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()

