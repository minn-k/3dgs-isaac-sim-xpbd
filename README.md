# 3DGS → OpenUSD → Isaac Sim: GPU XPBD Runtime

A research prototype for turning a reconstructed 3D Gaussian Splatting model into an interactive digital-twin scene. It connects a graph-based CUDA deformation solver to an OpenUSD scene in NVIDIA Isaac Sim and renders the deformed Gaussians through Fabric.

## Demonstration

Two Franka arms select Gaussian groups at opposite ends of a Wolf model, grasp them, pull outward, twist around the line between the grippers, return, and release. The public runtime also exposes a control window for automatic and manual two-arm commands.

This is a geometry-interaction prototype. A grasp attaches selected Gaussian groups to gripper targets; it is not a force-accurate tactile grasp simulator.

## Runtime data flow

~~~text
One-time setup
3DGS PLY + graph ── CPU → GPU ── CUDA XPBD solver
        └──────── OpenUSD Gaussian Splat scene

Every physics step
PhysX robot and collision pose ── CPU → GPU ── solver step

Every rendered frame
deformed position / orientation / scale ── GPU → CPU ── Fabric ── Isaac 3DGS renderer
~~~

The solver state stays on GPU between steps. The current bridge copies robot inputs to GPU at every physics step and reads deformed Gaussian attributes back to CPU once per changed rendered frame before writing Fabric attributes. See [the architecture note](docs/architecture.md) for the exact boundary.

## Quick start

1. Install Isaac Sim 6.1 and a compatible NVIDIA CUDA Toolkit. On first use, Isaac Sim must be able to download the Franka asset from NVIDIA Omniverse content, unless that asset is already cached or redirected through your configured asset source.
2. Build the Windows DLL from source using [the build guide](docs/build.md), or add a matching Windows release asset to the bin folder.
3. Put a redistributable Wolf asset bundle in examples/wolf. The source repository intentionally excludes generated PLY, NPZ, and USD splat assets. See [the data policy](docs/data.md).
4. Set ISAAC_SIM_ROOT and start Isaac Sim:

~~~bat
scripts\launch_isaac_safe.bat
~~~

5. In Isaac Sim, open examples/wolf/wolf_scene.usda.
6. In Window → Script Editor, replace the path below with your clone path and execute it:

~~~python
import os
os.environ["APG_3DGS_ISAAC_ROOT"] = r"C:\path\to\3dgs-isaac-sim-xpbd"
exec(open(os.path.join(os.environ["APG_3DGS_ISAAC_ROOT"], "src", "run_duo.py"), encoding="utf-8").read())
~~~

7. Press Play. Select Manual in the control window to move one or both grippers, grasp, pull, twist, and release.

## Repository map

~~~text
src/       Isaac runtime, CUDA bridge, collision conversion, and two-arm control
native/    C API bridge and CUDA solver source used to build the DLL
tools/     3DGS preparation, graph-order conversion, and OpenUSD conversion
examples/  light scene metadata and asset-bundle instructions
docs/      architecture, setup, build, data policy, and media policy
~~~

## Scope and reproducibility

- The public runtime supports the two-arm Wolf interaction workflow.
- The repository does not contain raw scans, faces, training images, learned 3DGS models, generated PLY, graph, or splat USD assets, compiled DLLs, or Wolf-derived screenshots.
- The native source is included so the Windows DLL can be rebuilt from a clean checkout.
- Fabric is used as a CPU-written runtime attribute interface in this prototype. It is not a zero-copy CUDA-to-Fabric path.

## Related work

apg-gs-chainmail will be released separately for the earlier ChainMail-focused viewer and paper scope. This repository contains the later OpenUSD, Isaac Sim, CUDA XPBD, and two-arm interaction integration.
