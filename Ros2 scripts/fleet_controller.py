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
from geometry_msgs.msg import Twist
from ackermann_msgs.msg import AckermannDriveStamped

import numpy as np
import math

class FleetController(Node):
    def __init__(self):
        super().__init__('fleet_controller')

        # ------Publishers------
        # lift
        self.pub_fl1_lift = self.create_publisher(Twist, '/forklift_1/lift', 10)
        self.pub_fl2_lift = self.create_publisher(Twist, '/forklift_2/lift', 10)
        self.pub_fl3_lift = self.create_publisher(Twist, '/forklift_3/lift', 10)

        # drive(ackermann)
        self.pub_fl1_drive = self.create_publisher(AckermannDriveStamped, '/forklift_1/ackermann', 10)
        self.pub_fl2_drive = self.create_publisher(AckermannDriveStamped, '/forklift_2/ackermann', 10)
        self.pub_fl3_drive = self.create_publisher(AckermannDriveStamped, '/forklift_3/ackermann', 10)

        self.get_logger().info("Fleet Controller is online. Ready to command the forklifts.")

        # Timer
        self.timer = self.create_timer(0.1, self.test_drive_callback)
        self.timer_lift = self.create_timer(0.5, self.test_lift_callback)
    
    def test_drive_callback(self):
        msg = AckermannDriveStamped()
        msg.drive.speed = 60.0
        msg.drive.steering_angle = 30.0

        self.pub_fl1_drive.publish(msg)
        self.get_logger().info("Driving Forklift 1 Forward", once=True)
        self.pub_fl2_drive.publish(msg)
        self.get_logger().info("Driving Forklift 2 Forward", once=True)
        self.pub_fl3_drive.publish(msg)
        self.get_logger().info("Driving Forklift 3 Forward", once=True)

    def test_lift_callback(self):
        msg = Twist()
        msg.linear.z = 2.0

        self.pub_fl1_lift.publish(msg)
        self.get_logger().info("Lifting Forklift 1 Up", once=True)
        self.pub_fl2_lift.publish(msg)
        self.get_logger().info("Lifting Forklift 2 Up", once=True)
        self.pub_fl3_lift.publish(msg)
        self.get_logger().info("Lifting Forklift 3 Up", once=True)

def main(args=None):
    rclpy.init(args=args)
    fleet_controller = FleetController()
    rclpy.spin(fleet_controller)
    fleet_controller.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()

