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


class ForkLift_1(Node):
    def __init__(self):
        super().__init__('forklift_1')

        # ------Subscribers------

        self.vision_subscription = self.create_subscription(
            Pose2D,
            '/forklift_1/target_pose',
            self.vision_callback,
            10
        )
        
        # ----------------------

        # ------Publishers------
        # lift
        self.pub_lift = self.create_publisher(Twist, '/forklift_1/lift', 10)

        # drive(ackermann)
        self.pub_drive = self.create_publisher(AckermannDriveStamped, '/forklift_1/ackermann', 10)
        # ----------------------


        self.get_logger().info("Forklift 1 is online. Ready to receive vision data and command the lift and drive.")


    def vision_callback(self, msg):
        # Extract target pose from the message
        cur_x = msg.x
        cur_y = msg.y
        cur_face = msg.theta

        self.get_logger().info(f"Received vision data: Target Pose - x: {cur_x}, y: {cur_y}, theta: {cur_face}")

        # Here you would implement your control logic to command the lift and drive based on the target pose.
        # For demonstration, we will just publish some dummy commands.

        # Example lift command (lift up)
        lift_cmd = Twist()
        lift_cmd.linear.z = 0.5  # Lift up at a speed of 0.5 m/s
        self.pub_lift.publish(lift_cmd)

        # Example drive command (move forward)
        drive_cmd = AckermannDriveStamped()
        drive_cmd.drive.speed = 1.0  # Move forward at 1 m/s
        drive_cmd.drive.steering_angle = 0.0  # No steering
        self.pub_drive.publish(drive_cmd)

    def PID_controller_lift(self, target_height, current_height):
        # Implement a simple PID controller for the lift
        Kp = 1.0  # Proportional gain
        Ki = 0.0  # Integral gain
        Kd = 0.0  # Derivative gain

        error = target_height - current_height
        control_signal = Kp * error  # + Ki * integral + Kd * derivative

        return control_signal
    
    def PID_controller_drive(self, target_pose, current_pose):
        # Implement a simple PID controller for the drive
        Kp = 1.0  # Proportional gain
        Ki = 0.0  # Integral gain
        Kd = 0.0  # Derivative gain

        error_x = target_pose.x - current_pose.x
        error_y = target_pose.y - current_pose.y
        error_theta = target_pose.theta - current_pose.theta

        control_signal_speed = Kp * math.sqrt(error_x**2 + error_y**2)  # Speed control based on distance
        control_signal_steering = Kp * error_theta  # Steering control based on angle error

        return control_signal_speed, control_signal_steering