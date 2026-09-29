<h1 align="center">3DGS → OpenUSD → Isaac Sim</h1>

<p align="center"><strong>GPU XPBD runtime for interactive 3D Gaussian Splatting scenes</strong></p>

<p align="center">
  <a href="https://minn-k.github.io/3d-representation-portfolio/"><img src="https://img.shields.io/badge/Portfolio-Website-green?logo=googlechrome&logoColor=white" alt="Portfolio"></a>
  <a href="https://github.com/minn-k/apg-gs-chainmail"><img src="https://img.shields.io/badge/Earlier_work-APG--GS_ChainMail-blue?logo=github" alt="APG-GS ChainMail"></a>
  <img src="https://img.shields.io/badge/Platform-Windows_11-0078D6?logo=windows&logoColor=white" alt="Windows 11">
  <img src="https://img.shields.io/badge/Runtime-NVIDIA_Isaac_Sim-76B900?logo=nvidia&logoColor=white" alt="NVIDIA Isaac Sim">
</p>

## Overview

This research prototype turns a reconstructed **3D Gaussian Splatting (3DGS)** scene into an interactive digital-twin
asset. A CUDA XPBD solver deforms a Gaussian graph, OpenUSD carries the scene into NVIDIA Isaac Sim, and Fabric writes
the updated Gaussian attributes for rendering. The focus is not force-accurate grasping; it is a practical bridge
between robot interaction, GPU deformation, and a Gaussian renderer.

## Highlights

- **Two-arm interaction.** Two Franka arms select Gaussian groups on a Wolf asset, grasp, pull, twist, return, and
  release them through an Isaac Sim control panel.
- **Gaussian-aware deformation.** The runtime updates position, orientation, and scale, rather than moving only
  Gaussian centres.
- **GPU-resident solver state.** Graph constraints and XPBD state remain on the GPU between physics steps.
- **OpenUSD / Fabric integration.** A generated OpenUSD scene hosts the Gaussian splat; deformed attributes are
  written into the Isaac-side rendering path.
- **Public source path.** The CUDA solver and bridge are included. Large reconstructed assets, generated PLY files,
  and binaries are intentionally excluded.

## System flow

~~~text
One-time preparation
3DGS PLY + Gaussian graph ── CPU → GPU ── CUDA XPBD solver
        └──────── OpenUSD Gaussian-splat scene

Interactive loop
PhysX gripper pose / collision input ── CPU → GPU ── solver step
deformed position / orientation / scale ── GPU → CPU ── Fabric ── Isaac 3DGS renderer
~~~

The current prototype keeps the solver state on GPU, copies robot input to GPU at each physics step, and reads updated
Gaussian attributes back once per changed render frame. See [docs/architecture.md](docs/architecture.md) for the exact
CPU/GPU boundary and data ownership.

## Quick start

### Requirements

- Windows 11, NVIDIA GPU, NVIDIA Isaac Sim 6.1, and a compatible CUDA Toolkit
- Visual Studio C++ Build Tools and CMake for the native DLL
- A redistributable 3DGS asset bundle. The included workflow uses the Wolf scene described in
  [examples/wolf/README.md](examples/wolf/README.md).

### Run the supplied workflow

1. Build the Windows DLL with [docs/build.md](docs/build.md).
2. Put the prepared Wolf asset bundle in <code>examples/wolf/</code> as described in [docs/data.md](docs/data.md).
3. Set <code>ISAAC_SIM_ROOT</code>, then start Isaac Sim:

   ~~~bat
   scripts\launch_isaac_safe.bat
   ~~~

4. Open <code>examples/wolf/wolf_scene.usda</code> in **Window → Script Editor**, replace the path and run:

   ~~~python
   import os
   os.environ["APG_3DGS_ISAAC_ROOT"] = r"C:\path\to\3dgs-isaac-sim-xpbd"
   exec(open(os.path.join(os.environ["APG_3DGS_ISAAC_ROOT"], "src", "run_duo.py"), encoding="utf-8").read())
   ~~~

5. Press Play. The control window provides automatic and manual commands for selecting, grasping, pulling, twisting,
   and releasing Gaussian groups.

## Repository map

~~~text
src/       Isaac runtime, CUDA bridge, collision conversion, and two-arm controls
native/    C API bridge and CUDA XPBD solver source
tools/     3DGS preparation, graph-order conversion, and OpenUSD conversion
examples/  lightweight scene metadata and asset-bundle instructions
docs/      setup, build, architecture, data, and media-policy notes
~~~

## Release scope

This is a focused research runtime, not a general-purpose robot grasp simulator.

- The public workflow supports the two-arm Wolf interaction path.
- No raw scans, training images, learned 3DGS models, generated PLY/NPZ/USD splats, compiled DLLs, or Wolf-derived
  screenshots are tracked.
- Fabric is used as a CPU-written attribute interface in this implementation; it is not a zero-copy CUDA-to-Fabric
  path.
- The native source is included so a compatible Windows DLL can be rebuilt from a clean checkout.

## Related projects

- [APG-GS ChainMail](https://github.com/minn-k/apg-gs-chainmail): the earlier SIBR Gaussian-viewer deformation overlay.
- [Editable Generative 3D Gaussians](https://github.com/minn-k/3d-representation-portfolio): generative 3D assets,
  semantic parts, and portfolio demonstrations.

## License and attribution

This repository contains derivative components from SIBR and 3D Gaussian Splatting. See [LICENSE.md](LICENSE.md),
[LICENSES](LICENSES), and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md); retain their notices when redistributing
modifications.
