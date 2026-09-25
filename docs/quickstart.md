# Quick start

## Prerequisites

- Windows 10 or 11 with an NVIDIA GPU
- Isaac Sim 6.1
- A CUDA build of xpbd_isaac.dll in the bin folder
- A Wolf asset bundle whose redistribution terms permit your use
- Internet access for the first Franka asset load, or an Isaac Sim cache or asset source that already provides that Franka USD

Build the DLL from source with [build.md](build.md), or add a matching release asset to bin\xpbd_isaac.dll.

The offline data-preparation tools use NumPy and SciPy. Install them in the Python environment used for those tools:

~~~bat
python -m pip install -r requirements-tools.txt
~~~

Isaac Sim itself provides the Omni, PXR, and Fabric modules; do not install those packages with pip.

## Asset bundle

The Wolf scene needs these files:

~~~text
examples/wolf/
  wolf_crop.ply
  wolf_graph.npz
  wolf_splat.usd
  wolf_scene.usda
  wolf_transform.json
~~~

The scene metadata and transform JSON are versioned here. The generated PLY, graph, and splat USD files are intentionally excluded. Do not combine a graph and PLY created from different crops: the runtime checks their rest-position ordering.

## Run in Isaac Sim

1. Set ISAAC_SIM_ROOT to the Isaac Sim installation folder.
2. Start Isaac Sim using scripts\launch_isaac_safe.bat.
3. Open examples/wolf/wolf_scene.usda.
4. Open Window → Script Editor and run:

~~~python
import os
os.environ["APG_3DGS_ISAAC_ROOT"] = r"C:\path\to\3dgs-isaac-sim-xpbd"
exec(open(os.path.join(os.environ["APG_3DGS_ISAAC_ROOT"], "src", "run_duo.py"), encoding="utf-8").read())
~~~

5. Press Play after the Wolf is settled on the floor.

The launcher creates a user-local cache directory and never deletes it automatically.
