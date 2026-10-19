# CENTRALIZED FORKLIFT SWARM CONTROLLER
This project simulates 3 forklifts working togather in stacking pallets using following technologies
- Issacsim
- Ros2
- Python

## Heading angle (theta)

The heading is an **absolute angle in the overhead camera image**. It does not depend on where the
forklift started or which way it faced before.

### How it is calculated (`Ros2 scripts/vision_tracker.py`)

For each ArUco tag, OpenCV returns the four corners in order: `c0` top-left, `c1` top-right,
`c2` bottom-right, `c3` bottom-left of the marker image.

1. **Centre** = midpoint of `c0` and `c2`.
2. **Front point** = midpoint of the top edge (`c0`–`c1`).
3. Heading = direction from the centre to the front point:
   ```python
   dx = front_x - center_x
   dy = center_y - front_y      # flipped, because image y points down
   heading = atan2(dy, dx)      # radians
   ```
4. It is smoothed with a low-pass filter (`alpha = 0.3`) on the wrapped angle difference, so it never
   jumps when crossing ±180°.
5. It is published as `theta` (radians) on `/forklift_N/vision_data`, with the tag centre `x, y` in pixels.

### What it is relative to

* **0° = image right** (+x in pixels). Angles increase **counter-clockwise** in the camera image.
* The overhead camera looks straight down with no rotation, so image right = world +X and image up =
  world +Y. `theta` is therefore also the yaw in the warehouse world frame.
* The forklift tags' top edge points along the **forks**, so `theta` = the direction the forks point.
  (The RL controller uses `tag_heading_offset_deg = 0` for this reason.)

```
                 90° (image up)
                      ↑
   ±180° (left) ←  [tag]  → 0° (image right)
                      ↓
                -90° (image down)
```

### Range

| Where | Units | Range |
|---|---|---|
| `vision_data.theta` (tracker output) | radians | −π … +π (−180° … +180°) |
| `lift_target_pos.theta` (goal you send) | **degrees** | any value, wrapped by the node (`270` = `-90`); `NaN` = any heading |
| Heading error inside the controller | radians | wrapped to −180° … +180° (always the short way round) |
| Fleet `status` display | degrees | −180° … +180° |

The tracker publishes **radians**; goals are sent in **degrees**.

### Pallets

Pallet tags use the same calculation and convention. On the warehouse pallets the tag's top edge points
perpendicular to a 1200 mm side, the side the forks can enter. The fleet `pallet` command approaches along
`theta` or `theta + 180°`, whichever face looks toward the forklift.

### Example

`{x: 900, y: 360, theta: 90.0}` on `/forklift_1/lift_target_pos`: put the tag at pixel (900, 360) with the
forks pointing toward the top of the camera image (world +Y).
