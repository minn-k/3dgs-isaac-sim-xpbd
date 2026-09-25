"""Binary PLY I/O for the 3D Gaussian Splatting assets used by this runtime."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def read_ply(path: str | Path):
    """Read a binary-little-endian, float-property vertex PLY file."""
    path = Path(path)
    header: list[str] = []
    with path.open("rb") as stream:
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"PLY header did not terminate: {path}")
            item = line.decode("ascii").rstrip("\r\n")
            header.append(item)
            if item == "end_header":
                break
        offset = stream.tell()

    if "format binary_little_endian 1.0" not in header:
        raise ValueError("Only binary_little_endian PLY files are supported.")

    count = None
    names: list[str] = []
    for item in header:
        fields = item.split()
        if not fields:
            continue
        if fields[0] == "element":
            if len(fields) != 3 or fields[1] != "vertex" or count is not None:
                raise ValueError(f"Expected one vertex element, got: {item}")
            count = int(fields[2])
        elif fields[0] == "property":
            if len(fields) != 3 or fields[1] != "float":
                raise ValueError(f"Only float properties are supported: {item}")
            names.append(fields[2])

    if count is None:
        raise ValueError("PLY header does not define a vertex element.")
    data = np.fromfile(path, dtype=np.dtype([(name, "<f4") for name in names]), count=count, offset=offset)
    if len(data) != count:
        raise ValueError(f"Expected {count} vertices, read {len(data)}.")
    return header, data


def write_ply(path: str | Path, header: list[str], data: np.ndarray) -> None:
    """Write a PLY with a preserved header and updated vertex count."""
    path = Path(path)
    output_header = [
        f"element vertex {data.shape[0]}" if item.startswith("element vertex") else item
        for item in header
    ]
    with path.open("wb") as stream:
        stream.write(("\n".join(output_header) + "\n").encode("ascii"))
        stream.write(data.tobytes())