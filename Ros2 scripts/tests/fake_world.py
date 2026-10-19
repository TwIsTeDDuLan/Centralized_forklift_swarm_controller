"""Kinematic stand-in for Isaac Sim + vision_tracker + fork lidar: forklift 1 and pallet 1.

Forklift: rear-steer bicycle (signs measured in Isaac Lab), tag theta = fork heading (low-pass 0.3).
Pallet: EUR box model from ForkliftInsertEnv._lidar_boxes, tag theta = normal of a 1200 mm face.
Mast: /forklift_1/lift target position, moves at 0.3 m/s, reported on /forklift_1/mast_pos.
Pixels: u = 640 + 65 x, v = 360 - 65 y.
"""
import math
import sys

sys.path.append('/opt/ros/jazzy/lib/python3.12/site-packages')
import numpy as np
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Pose2D, Twist
from sensor_msgs.msg import JointState, LaserScan

PX, U0, V0 = 65.0, 640.0, 360.0
L = 1.39
TAG_FROM_FRONT = 1.19   # tag ahead of roller axle along body +X
TAG_TO_TIP = 2.518      # tag to fork tips along the forks (body -X)
TIP_HALF = 0.304        # tines at +-0.304 m
PALLET = (3.0, -1.0)    # world centre
PALLET_NORMAL = math.pi  # tag top edge (a 1200 mm face normal), tracker convention


def to_px(x, y):
    return U0 + PX * x, V0 - PX * y


class World(Node):
    def __init__(self):
        super().__init__('fake_world')
        self.fx, self.fy, self.yaw = (float(v) for v in (sys.argv[1:4] if len(sys.argv) > 3 else (-7.5, 2.5, 90.0)))
        self.yaw = math.radians(self.yaw)
        self.speed = self.steer = 0.0
        self.q = -0.146
        self.q_target = -0.146
        self.vis_theta = None
        self.create_subscription(AckermannDriveStamped, '/forklift_1/ackermann', self.cmd, 10)
        self.create_subscription(Twist, '/forklift_1/lift', self.lift_cmd, 10)
        self.pub_vis = self.create_publisher(Pose2D, '/forklift_1/vision_data', 10)
        self.pub_pal = self.create_publisher(Pose2D, '/pallet_1/vision_data', 10)
        self.pub_scan = self.create_publisher(LaserScan, '/forklift_1/scan', 10)
        self.pub_mast = self.create_publisher(JointState, '/forklift_1/mast_pos', 10)
        self.create_timer(1 / 60, self.step)
        self.create_timer(1 / 30, self.publish)
        self.create_timer(0.1, self.scan)
        self.create_timer(1.0, self.log)
        el = np.linspace(-math.radians(8), math.radians(8), 3)
        az = np.linspace(-math.radians(45), math.radians(45), 64)
        self.el, self.az = [a.reshape(-1) for a in np.meshgrid(el, az, indexing='ij')]

    def cmd(self, m):
        self.speed, self.steer = m.drive.speed, max(-60.0, min(60.0, m.drive.steering_angle))

    def lift_cmd(self, m):
        self.q_target = max(-0.15, min(2.0, m.linear.z))

    def step(self):
        dt = 1 / 60
        vw = math.radians(self.speed) * 0.16
        d = math.radians(self.steer)
        u, w = vw * math.cos(d), -vw * math.sin(d) / L
        self.fx += u * math.cos(self.yaw) * dt
        self.fy += u * math.sin(self.yaw) * dt
        self.yaw += w * dt
        self.q += max(-0.3 * dt, min(0.3 * dt, self.q_target - self.q))

    def tag(self):
        return self.fx + TAG_FROM_FRONT * math.cos(self.yaw), self.fy + TAG_FROM_FRONT * math.sin(self.yaw)

    def fork_dir(self):
        return self.yaw + math.pi

    def entry_frame(self):
        """Pallet frame for the face the forks point at: origin at face centre, +X into the pallet."""
        f = self.fork_dir()
        h = PALLET_NORMAL if math.cos(PALLET_NORMAL - f) >= 0 else PALLET_NORMAL + math.pi
        ox, oy = PALLET[0] - 0.4 * math.cos(h), PALLET[1] - 0.4 * math.sin(h)
        return ox, oy, h

    def to_pallet(self, x, y, h0):
        ox, oy, h = h0
        dx, dy = x - ox, y - oy
        return math.cos(h) * dx + math.sin(h) * dy, -math.sin(h) * dx + math.cos(h) * dy

    def tips(self):
        tx, ty = self.tag()
        f = self.fork_dir()
        cx, cy = tx + TAG_TO_TIP * math.cos(f), ty + TAG_TO_TIP * math.sin(f)
        lx, ly = -math.sin(f), math.cos(f)  # left of the fork direction
        return [(cx + s * TIP_HALF * lx, cy + s * TIP_HALF * ly) for s in (1, -1)]

    def publish(self):
        tx, ty = self.tag()
        theta = math.atan2(math.sin(self.fork_dir()), math.cos(self.fork_dir()))
        if self.vis_theta is None:
            self.vis_theta = theta
        d = math.atan2(math.sin(theta - self.vis_theta), math.cos(theta - self.vis_theta))
        self.vis_theta = math.atan2(math.sin(self.vis_theta + 0.3 * d), math.cos(self.vis_theta + 0.3 * d))
        u, v = to_px(tx, ty)
        self.pub_vis.publish(Pose2D(x=float(int(u)), y=float(int(v)), theta=self.vis_theta))
        pu, pv = to_px(*PALLET)
        self.pub_pal.publish(Pose2D(x=float(int(pu)), y=float(int(pv)), theta=PALLET_NORMAL))
        self.pub_mast.publish(JointState(name=['lift_joint'], position=[self.q]))

    def scan(self):
        fr = self.entry_frame()
        tx, ty = self.tag()
        f = self.fork_dir()
        ow = (tx + (TAG_TO_TIP - 0.138) * math.cos(f), ty + (TAG_TO_TIP - 0.138) * math.sin(f))
        ox, oy = self.to_pallet(*ow, fr)
        fl = f - fr[2]
        fwd = np.array([math.cos(fl), math.sin(fl), 0.0])
        right = np.array([math.sin(fl), -math.cos(fl), 0.0])
        up = np.array([0.0, 0.0, 1.0])
        dirs = (np.cos(self.el)[:, None] * np.cos(self.az)[:, None] * fwd + np.cos(self.el)[:, None] * np.sin(self.az)[:, None] * right
                + np.sin(self.el)[:, None] * up)
        o = np.array([ox, oy, self.q + 0.28])
        boxes = [((0, 0.05), (y - 0.05, y + 0.05), (0, 0.144)) for y in (-0.55, 0.0, 0.55)]
        boxes += [((0, 0.8), (-0.6, 0.6), (0, 0.02)), ((0, 0.8), (-0.6, 0.6), (0.10, 0.144)),
                  ((0.76, 0.8), (-0.6, 0.6), (0, 0.144))]
        ranges = np.full(len(dirs), 4.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            for bx in boxes:
                lo = np.array([b[0] for b in bx]); hi = np.array([b[1] for b in bx])
                t1, t2 = (lo - o) / dirs, (hi - o) / dirs
                tn = np.nanmax(np.minimum(t1, t2), axis=1)
                tf = np.nanmin(np.maximum(t1, t2), axis=1)
                hit = (tf >= np.maximum(tn, 0)) & (tn < ranges)
                ranges = np.where(hit, np.maximum(tn, 0), ranges)
        self.pub_scan.publish(LaserScan(ranges=[float(r) for r in ranges], range_max=4.0))

    def log(self):
        fr = self.entry_frame()
        tips = [self.to_pallet(x, y, fr) for x, y in self.tips()]
        tx, ty = self.tag()
        print(f"TRUE tag px={tuple(int(c) for c in to_px(tx, ty))} fork={math.degrees(self.fork_dir()) % 360:.0f} "
              f"lift={self.q:.2f}  tips in pallet frame (depth, lateral)="
              f"({tips[0][0]:.2f}, {tips[0][1]:+.2f}) ({tips[1][0]:.2f}, {tips[1][1]:+.2f})", flush=True)


rclpy.init()
rclpy.spin(World())
