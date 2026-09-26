"""Script Editor entry point for the public two-arm Wolf interaction demo.

Set APG_3DGS_ISAAC_ROOT to this repository root before executing this file
from Isaac Sim's Script Editor. The wrapper gives runtime_loop.py a stable
__file__ path and forces the supported public demo mode.
"""
from __future__ import annotations

import os
from pathlib import Path

root = os.environ.get("APG_3DGS_ISAAC_ROOT")
if not root:
    raise RuntimeError(
        "Set APG_3DGS_ISAAC_ROOT to the repository root before running this file. "
        "See docs/quickstart.md."
    )

repo_root = Path(root).expanduser().resolve()
runtime = repo_root / "src" / "runtime_loop.py"
if not runtime.is_file():
    raise FileNotFoundError(f"Could not find runtime script: {runtime}")

scope = {
    "__name__": "__main__",
    "__file__": str(runtime),
}
exec(compile(runtime.read_text(encoding="utf-8"), str(runtime), "exec"), scope)