"""EUR pallet geometry and Isaac Lab spawn configs for the insertion task.

`Issasim Models/o3dyn_pallet.usd` (`/pallet_loaded`) is used as a visual only. Its authored
root scale is non-uniform (wood world AABB ~1.00 x 2.31 x 0.33 m) and its `collision_boxes`
cube wraps the euro-case load, so it is not used for contact or lidar. Measured 2026-10-18:

  * Wood mesh local points are a 79.6 x 14.0 x 120.0 cm EUR deck (X = 800 mm width,
    Y = height, Z = 1200 mm length).
  * The 1200 mm faces (local ±X) have two openings ~45 cm wide, centers at local Z ±27.5 cm,
    opening height local Y ≈ 2–10 cm. Those match forklift_b tines at ±30.4 cm, 7 cm wide.
  * The 800 mm faces do not fit the 59 cm tine spacing, so insertion is through a 1200 mm face.

Canonical pallet frame used by the env (after spawn alignment):
  * origin on the floor at the entry-face centre
  * +X into the pallet (insertion, depth 0.80 m)
  * +Y left along the 1.20 m face
  * +Z up
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sim.spawners.materials import UsdPhysicsRigidBodyMaterialCfg
from isaaclab.utils import configclass

O3DYN_PALLET_USD = Path(__file__).resolve().parents[2] / "Issasim Models" / "o3dyn_pallet.usd"

# ------ canonical EUR deck (metres) ------
PALLET_DEPTH = 0.80  # insertion axis (800 mm)
PALLET_WIDTH = 1.20  # face width (1200 mm)
PALLET_HEIGHT = 0.144  # deck + stack pitch
# Two openings on the 1200 mm face (outer pair). Width is the measured ~45 cm gap, slightly
# tightened so a tine that clips a stringer counts as a miss.
POCKET_Y = (-0.275, 0.275)
POCKET_WIDTH = 0.40
POCKET_HEIGHT = 0.08
POCKET_Z_LO = 0.02
POCKET_Z_HI = 0.10
POCKET_Z_CENTER = 0.5 * (POCKET_Z_LO + POCKET_Z_HI)
# Reward / success target: near the far end, at the forklift's native tine spacing.
# Order is (left tine, right tine) in the pallet frame at the insertion heading (body yaw ≈ π,
# forks along +X). Lift +X (FORK_TIP_LEFT) maps to pallet +Y; lift −X maps to pallet −Y.
INSERT_TARGET_X = 0.70
INSERT_TARGET_Y = (0.304, -0.304)
MAX_STACK = 3

# Visual-only correction so the distorted o3dyn wood AABB becomes 0.80 x 1.20 x 0.144,
# with the 1200 mm face at x = 0, y centred, z = 0. See scripts/measure_geometry.py.
O3DYN_VISUAL_SCALE = (0.798, 0.520, 0.436)
O3DYN_VISUAL_OFFSET = (0.192, 0.092, -0.001)

# Staging: fork tips this far in front of the face before the insert policy takes over.
STAGING_TIP_STANDOFF = 1.5


def _kinematic_cuboid(size, color, *, visible=True) -> sim_utils.CuboidCfg:
    return sim_utils.CuboidCfg(
        size=size,
        rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
        mass_props=sim_utils.MassCfg(mass=1.0),
        collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        physics_material=UsdPhysicsRigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        visible=visible,
    )


@configclass
class PalletSceneAssetsCfg:
    """Visual o3dyn pallet + kinematic face boards / extra decks. Pallet stays fixed in the env frame."""

    pallet_visual = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PalletVisual",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(O3DYN_PALLET_USD),
            scale=O3DYN_VISUAL_SCALE,
            collision_props=sim_utils.UsdPhysicsCollisionCfg(collision_enabled=False),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=O3DYN_VISUAL_OFFSET),
    )
    # Three entry-face stringers (contact with these + tips not in a pocket = face hit).
    face_left: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/FaceLeft",
        spawn=_kinematic_cuboid((0.05, 0.10, PALLET_HEIGHT), (0.55, 0.22, 0.08)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.025, -0.55, 0.5 * PALLET_HEIGHT)),
    )
    face_center: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/FaceCenter",
        spawn=_kinematic_cuboid((0.05, 0.10, PALLET_HEIGHT), (0.55, 0.22, 0.08)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.025, 0.0, 0.5 * PALLET_HEIGHT)),
    )
    face_right: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/FaceRight",
        spawn=_kinematic_cuboid((0.05, 0.10, PALLET_HEIGHT), (0.55, 0.22, 0.08)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.025, 0.55, 0.5 * PALLET_HEIGHT)),
    )
    # Extra stacked empty decks (moved underground when unused).
    stack_1: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Stack1",
        spawn=_kinematic_cuboid((PALLET_DEPTH, PALLET_WIDTH, PALLET_HEIGHT), (0.65, 0.40, 0.18)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.40, 0.0, -2.0)),
    )
    stack_2: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Stack2",
        spawn=_kinematic_cuboid((PALLET_DEPTH, PALLET_WIDTH, PALLET_HEIGHT), (0.65, 0.40, 0.18)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.40, 0.0, -2.0)),
    )


def staging_tag_distance(tip_standoff: float = STAGING_TIP_STANDOFF) -> float:
    """Distance from the pallet face to the forklift *tag* when the tips are `tip_standoff` out."""
    from .robot import TAG_TO_TIP

    return tip_standoff + TAG_TO_TIP
