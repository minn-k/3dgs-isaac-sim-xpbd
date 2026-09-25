# Building `xpbd_isaac.dll` on Windows

## Requirements

- Windows 10/11 and an NVIDIA GPU
- CMake 3.22 or newer
- CUDA Toolkit with `nvcc`
- Visual Studio C++ build tools. Set `CMAKE_GENERATOR` if your installed generator differs.

The solver core is included under `native/solver_core/` so the native build does not require a local copy of the original working directory.

## Build

```bat
set CUDA_ARCHS=86
set CMAKE_GENERATOR=Visual Studio 17 2022
scripts\build_windows.bat
```

For an RTX 40-series GPU, set `CUDA_ARCHS=89`; use the CUDA architecture matching your target GPU. The resulting file is written to `bin\xpbd_isaac.dll`.

## Release binary policy

Do not commit `.dll`, `.lib`, `.exp`, or `.pdb` files to Git. Publish an optional Windows binary as a GitHub Release asset with its CUDA architecture and SHA-256 checksum in the asset name or release notes.
