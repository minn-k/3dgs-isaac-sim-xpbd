# Runtime architecture

The system keeps rendering, scene representation, and deformation solving separate, then connects them once per frame.

## Initialization

1. A prepared 3D Gaussian model supplies rest positions, activated scales, rotations, opacities, and graph edges.
2. `xpbd_bridge.py` loads `xpbd_isaac.dll` and sends the static solver inputs to GPU memory once.
3. `wolf_scene.usda` references the Gaussian Splat USD object and places it in a meter-scale OpenUSD scene.

## Repeated runtime loop

1. Isaac PhysX advances the Franka robots and collision objects.
2. `isaac_colliders.py` reads their current poses and converts them from Isaac world coordinates into the source 3DGS coordinate system.
3. The bridge passes collider poses and attachment targets to the CUDA solver. The CUDA solver advances the deformable Gaussian state once per PhysX step.
4. The bridge reads updated Gaussian positions, scales, and orientations from GPU memory into NumPy arrays.
5. `runtime_loop.py` writes those arrays to Fabric runtime attributes on the Splat prim. The Isaac 3DGS renderer displays the updated splats.
6. The next physics/render frame repeats the same sequence.

## Memory path

| Data | Direction | Frequency |
|---|---|---|
| Rest Gaussian state and graph | CPU → GPU | Once at solver creation |
| Robot/collider pose and grasp targets | CPU → GPU | Every physics step |
| Deformed position, scale, orientation | GPU → CPU | Once per rendered frame when state changed |
| Fabric attributes | CPU → Isaac renderer | Once per rendered frame when state changed |

`Fabric` is the real-time scene data layer that the Isaac renderer reads. The current prototype writes host arrays into Fabric attributes; it does not pass a CUDA device pointer directly to Fabric.
