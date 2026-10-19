"""One-shot USD measurement: pallet pockets and forklift fork tips / lift link.

Run with Isaac Sim's Python (has pxr):
    ~/issacsim/python.sh scripts/measure_geometry.py
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics

HERE = Path(__file__).resolve().parents[1]
MODELS = HERE.parent / "Issasim Models"


def world_bbox(prim, cache):
    try:
        b = cache.ComputeWorldBound(prim)
        r = b.ComputeAlignedRange()
        if r.IsEmpty():
            return None
        mn, mx = r.GetMin(), r.GetMax()
        return np.array(mn), np.array(mx)
    except Exception:
        return None


def mesh_world_points(prim):
    mesh = UsdGeom.Mesh(prim)
    pts = mesh.GetPointsAttr().Get()
    if not pts:
        return None
    xform = UsdGeom.Xformable(prim)
    m = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    out = []
    for p in pts:
        wp = m.Transform(p)
        out.append((wp[0], wp[1], wp[2]))
    return np.array(out, dtype=np.float64)


def dump_stage(path: Path, max_depth=8):
    stage = Usd.Stage.Open(str(path))
    print("=" * 80)
    print("FILE", path.name)
    print("upAxis", UsdGeom.GetStageUpAxis(stage), "metersPerUnit", UsdGeom.GetStageMetersPerUnit(stage))
    print("defaultPrim", stage.GetDefaultPrim().GetPath() if stage.GetDefaultPrim() else None)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy])
    for prim in stage.Traverse():
        depth = str(prim.GetPath()).count("/")
        if depth > max_depth:
            continue
        t = prim.GetTypeName()
        interesting = (
            t in ("Xform", "Mesh", "Cube", "Cylinder", "Sphere", "Capsule", "PhysicsJoint", "PhysicsPrismaticJoint")
            or any(k in str(prim.GetPath()).lower() for k in ("pallet", "fork", "lift", "collision", "wheel", "joint"))
        )
        if not interesting:
            continue
        bb = world_bbox(prim, cache)
        extra = ""
        if bb:
            mn, mx = bb
            extra = (
                f" bbox=({mn[0]:.3f},{mn[1]:.3f},{mn[2]:.3f})-({mx[0]:.3f},{mx[1]:.3f},{mx[2]:.3f})"
                f" size=({mx[0]-mn[0]:.3f},{mx[1]-mn[1]:.3f},{mx[2]-mn[2]:.3f})"
            )
        print(f"{'  '*min(depth, 8)}{prim.GetPath()} [{t}]{extra}")
    return stage


def analyze_pallet(stage):
    print("\n----- pallet wood mesh occupancy -----")
    wood = None
    for prim in stage.Traverse():
        name = prim.GetName().lower()
        if prim.GetTypeName() == "Mesh" and "palletwood" in name:
            wood = prim
            break
    if wood is None:
        print("no SM_PalletWood mesh")
        return
    pts = mesh_world_points(wood)
    print(f"wood prim {wood.GetPath()} npts={len(pts)}")
    print(f"  range xyz {pts.min(0)} .. {pts.max(0)}")
    print(f"  size {pts.max(0) - pts.min(0)}")

    # Occupancy along each axis at deck height (z in [0, 0.14])
    deck = pts[(pts[:, 2] >= -0.01) & (pts[:, 2] <= 0.16)]
    print(f"  deck points {len(deck)}")

    def occupancy(axis, lo, hi, bins=80, zmax=0.16):
        sel = pts[(pts[:, 2] >= -0.01) & (pts[:, 2] <= zmax)]
        hist, edges = np.histogram(sel[:, axis], bins=bins, range=(lo, hi))
        return hist, edges

    for axis, name, lo, hi in ((0, "X", -0.7, 0.7), (1, "Y", -0.5, 0.5)):
        hist, edges = occupancy(axis, lo, hi)
        print(f"\n  occupancy {name} (count of deck verts):")
        for i, c in enumerate(hist):
            if c > 0:
                print(f"    [{edges[i]:6.3f},{edges[i+1]:6.3f}) {c:5d} {'#' * min(c // 20, 40)}")

    # Slice the front faces: at x ~ ±0.6 and y ~ ±0.4, look at y/x gaps (pocket openings).
    print("\n  face slices (points near each face, z in 0.02..0.12 = pocket height):")
    pocket_z = pts[(pts[:, 2] > 0.02) & (pts[:, 2] < 0.12)]
    faces = {
        "+X": pocket_z[np.abs(pocket_z[:, 0] - pocket_z[:, 0].max()) < 0.04],
        "-X": pocket_z[np.abs(pocket_z[:, 0] - pocket_z[:, 0].min()) < 0.04],
        "+Y": pocket_z[np.abs(pocket_z[:, 1] - pocket_z[:, 1].max()) < 0.04],
        "-Y": pocket_z[np.abs(pocket_z[:, 1] - pocket_z[:, 1].min()) < 0.04],
    }
    for fname, fp in faces.items():
        if len(fp) < 10:
            print(f"    {fname}: {len(fp)} pts (too few)")
            continue
        axis = 1 if fname.endswith("X") else 0
        hist, edges = np.histogram(fp[:, axis], bins=40)
        print(f"    {fname}: {len(fp)} pts along {'Y' if axis==1 else 'X'}")
        for i, c in enumerate(hist):
            mark = "#" * min(c // 2, 30) if c else "."
            print(f"      [{edges[i]:6.3f},{edges[i+1]:6.3f}) {c:4d} {mark}")


def analyze_forks(stage):
    print("\n----- forklift lift / fork mesh -----")
    lift_mesh = None
    lift_xforms = []
    joints = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        t = prim.GetTypeName()
        if "lift" in path.lower() or "fork" in path.lower():
            if t == "Mesh" and lift_mesh is None:
                lift_mesh = prim
            lift_xforms.append((path, t))
        if "Joint" in t or prim.HasAPI(UsdPhysics.Joint):
            joints.append((path, t))
        # also Physics joints via applied API
        if prim.HasAttribute("physics:jointEnabled") or "joint" in path.lower():
            joints.append((path, t))

    print("lift/fork prims:")
    for p, t in lift_xforms:
        print(f"  {p} [{t}]")
    print("joint-ish prims:")
    seen = set()
    for p, t in joints:
        if p not in seen:
            print(f"  {p} [{t}]")
            seen.add(p)

    # Find lift joint local positions
    print("\nlift_joint attrs:")
    for prim in stage.Traverse():
        if prim.GetName() == "lift_joint":
            for attr in prim.GetAttributes():
                val = attr.Get()
                if val is not None:
                    print(f"  {attr.GetName()} = {val}")

    meshes = []
    for prim in stage.Traverse():
        if prim.GetTypeName() == "Mesh" and (
            "lift" in prim.GetName().lower() or "fork" in str(prim.GetPath()).lower()
        ):
            meshes.append(prim)
    if not meshes:
        # fallback: any mesh whose bbox reaches x < -1.5
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy])
        for prim in stage.Traverse():
            if prim.GetTypeName() != "Mesh":
                continue
            bb = world_bbox(prim, cache)
            if bb and bb[0][0] < -1.5:
                meshes.append(prim)

    for mesh in meshes:
        pts = mesh_world_points(mesh)
        if pts is None or len(pts) == 0:
            continue
        print(f"\nmesh {mesh.GetPath()} npts={len(pts)}")
        print(f"  range {pts.min(0)} .. {pts.max(0)}")
        # fork tines: most -X vertices, low z
        tips = pts[pts[:, 0] < pts[:, 0].min() + 0.08]
        print(f"  tip verts (x < xmin+8cm): {len(tips)} y range {tips[:,1].min():.3f}..{tips[:,1].max():.3f} z {tips[:,2].min():.3f}..{tips[:,2].max():.3f}")
        # cluster tips in Y to find two tines
        ys = tips[:, 1]
        hist, edges = np.histogram(ys, bins=40)
        print("  tip Y occupancy:")
        for i, c in enumerate(hist):
            if c:
                print(f"    [{edges[i]:6.3f},{edges[i+1]:6.3f}) {c:4d}")
        # tine height: verts with x < -1.5 and |y| in tine bands
        tine = pts[pts[:, 0] < -1.5]
        print(f"  x<-1.5 verts: {len(tine)} z {tine[:,2].min():.3f}..{tine[:,2].max():.3f}")
        histz, ez = np.histogram(tine[:, 2], bins=30)
        print("  tine Z occupancy (x<-1.5):")
        for i, c in enumerate(histz):
            if c:
                print(f"    [{ez[i]:6.3f},{ez[i+1]:.3f}) {c:4d}")


def main():
    pallet = dump_stage(MODELS / "o3dyn_pallet.usd", max_depth=10)
    analyze_pallet(pallet)
    fork = dump_stage(MODELS / "forklift_b.usd", max_depth=6)
    analyze_forks(fork)


if __name__ == "__main__":
    main()
