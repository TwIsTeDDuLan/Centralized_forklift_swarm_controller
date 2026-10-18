"""Isaac Lab articulation config for forklift_b.

Body frame of forklift_b.usd (root prim /SMV_Forklift_B01_01, Z up, meters):
  * forks and the four passive rollers are on the -X side (rollers at x = -0.91 / -0.72),
  * the driven + steered rear wheel (sphere, r = 0.16 m) is at x = +0.58, so the wheelbase is ~1.39 m,
  * the ArUco tag the overhead camera tracks sits on the mast at x ~ +0.38, y ~ 0.

Drive model, identical to the Forklift_ROS2.usd action graph so a policy can publish straight to
/forklift_N/ackermann:
  * AckermannDrive.speed          -> back_wheel_drive  velocity target [deg/s]
  * AckermannDrive.steering_angle -> back_wheel_swivel position target [deg], USD limit +-60
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg

# The USD stays in the project's model folder on the Local Disk; nothing is copied to the NVMe.
FORKLIFT_B_USD = Path(__file__).resolve().parents[2] / "Issasim Models" / "forklift_b.usd"

DRIVE_JOINT = "back_wheel_drive"
STEER_JOINT = "back_wheel_swivel"
LIFT_JOINT = "lift_joint"
ROLLER_JOINTS = [".*_roller"]

WHEEL_RADIUS = 0.16  # [m] back_wheel/Sphere
WHEELBASE = 1.39  # [m] rear wheel to the middle of the roller pairs
TAG_OFFSET_B = (0.38, 0.0)  # [m] ArUco tag position in the body frame (what vision_tracker.py reports)

FORKLIFT_B_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(FORKLIFT_B_USD),
        activate_contact_sensors=False,
        rigid_props=PhysxRigidBodyCfg(
            disable_gravity=False,
            max_linear_velocity=20.0,
            max_angular_velocity=50.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=PhysxArticulationCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=1
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.02),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        # Velocity-controlled drive wheel. The USD also authors stiffness=100 on this continuous joint, which
        # springs the wheel back toward 0 deg as it winds up; zero it here (see README, "Sim-to-sim notes").
        "drive": ImplicitActuatorCfg(joint_names_expr=[DRIVE_JOINT], stiffness=0.0, damping=None),
        # Position-controlled steering and mast: keep the gains authored in the USD (same as Isaac Sim).
        "steer": ImplicitActuatorCfg(joint_names_expr=[STEER_JOINT], stiffness=None, damping=None),
        "lift": ImplicitActuatorCfg(joint_names_expr=[LIFT_JOINT], stiffness=None, damping=None),
        # Free-spinning load rollers.
        "rollers": ImplicitActuatorCfg(joint_names_expr=ROLLER_JOINTS, stiffness=0.0, damping=0.0),
    },
)
"""forklift_b with USD drive gains; actions are joint targets in the same units as the ROS Ackermann topic."""
