#!/usr/bin/env python3

import os
import sys

# Add ROS Python packages path if the shell has not sourced ROS setup.
ros_python_path = '/opt/ros/jazzy/lib/python3.12/site-packages'
if os.path.isdir(ros_python_path) and ros_python_path not in sys.path:
    sys.path.append(ros_python_path)

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist, Pose2D
from ackermann_msgs.msg import AckermannDriveStamped

import math

class ForkLift_1(Node):
    def __init__(self):
        super().__init__('forklift_1')

        # ------Parameters (The "State") ------
        self.mast_height = 0.0
        self.target_mast_height = 0.0  # NEW: Store the requested goal
        self.prev_time_lift = 0
        self.integaral_sum_lift = 0
        self.prev_error_lift = 0

        # -------------------------------------

        # ------Subscribers------
        self.sub_target_pos = self.create_subscription(
            Pose2D,
            '/forklift_1/target_pos',
            self.target_pos_callback,
            10
        )

        # FIXED: Unique variable name
        self.sub_target_mast = self.create_subscription(
            Twist,
            '/forklift_1/mast_target_pos',
            self.target_mast_callback,
            10
        )

        # FIXED: Unique variable name
        self.sub_mast_state = self.create_subscription(
            JointState,
            '/forklift_1/mast_pos',
            self.mast_callback,
            10
        )
        # ----------------------

        # ------Publishers------
        self.pub_lift = self.create_publisher(Twist, '/forklift_1/lift', 10)
        self.pub_drive = self.create_publisher(AckermannDriveStamped, '/forklift_1/ackermann', 10)
        # ----------------------

        # ------Timers------
        # NEW: Create a control loop that runs at 20Hz (every 0.05 seconds)
        self.control_timer = self.create_timer(0.05, self.control_loop)
        # ------------------

        self.get_logger().info("Forklift 1 is online. Ready to receive vision data and command the lift and drive.")

    def target_pos_callback(self, msg: Pose2D):
        pass

    def target_mast_callback(self, msg: Twist):
        # 1. Update the state ONLY. Do not execute movement here.
        self.target_mast_height = msg.linear.z
        self.get_logger().info(f"New mast target received: {self.target_mast_height}")

    def mast_callback(self, msg: JointState):
        if msg.position:
            self.mast_height = msg.position[0]

    
    
    def control_loop(self):
        # 2. This runs continuously, calculating the PID based on the current state
        error = self.target_mast_height - self.mast_height
        speed = 0.1 # meters per second

        if abs(error) > 0.05:
            control_signal = self.mast_height + speed * 0.05 + 0.1 * error  # P control with feedforward
            # 4. Clamp the signal to safe limits (not reject it)
            control_signal = max(-0.15, min(2.0, control_signal))

            lift_cmd = Twist()
            lift_cmd.linear.z = control_signal
            self.pub_lift.publish(lift_cmd)
            self.get_logger().info(f"\nPublished lift command: {lift_cmd.linear.z} \nmast_cur_height:{self.mast_height} \ntarge_height:{self.target_mast_height}")

    def PID_controller_drive(self, target_pose, current_pose):
        Kp = 0.3
        Ki = 0.0
        Kd = 0.0

        error_x = target_pose.x - current_pose.x
        error_y = target_pose.y - current_pose.y
        error_theta = target_pose.theta - current_pose.theta

        control_signal_speed = Kp * math.sqrt(error_x**2 + error_y**2)
        control_signal_steering = Kp * error_theta

        return control_signal_speed, control_signal_steering

def main(args=None):
    rclpy.init(args=args)
    fleet_controller = ForkLift_1()
    rclpy.spin(fleet_controller)
    fleet_controller.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()