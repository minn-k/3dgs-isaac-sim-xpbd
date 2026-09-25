r"""
Convert a prepared 3D Gaussian Splatting asset into an Isaac Sim OpenUSD scene.

Run with Isaac Sim Python:
  %ISAAC_SIM_ROOT%\python.bat build_usd.py --dir <prepared asset directory> --name wolf

The tool converts <name>_crop.ply into <name>_splat.usd with the Isaac Gaussian
Splat converter, then writes <name>_scene.usda. The scene applies the transform
calculated by prepare_splat.py and references the splat USD below /World/<Name>/Splat.
"""
import argparse
import glob
import json
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--name", required=True)
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402  (pxr 은 Kit 가 뜬 뒤에 import 된다)

app = SimulationApp({"headless": True})
try:
    isaac_root = os.environ.get("ISAAC_SIM_ROOT") or os.environ.get("ISAAC_PATH")
    if not isaac_root:
        raise RuntimeError("Set ISAAC_SIM_ROOT to the Isaac Sim installation directory.")
    prebundle = sorted(glob.glob(os.path.join(isaac_root,
                                              "extscache", "omni.kit.converter.gsplat-*", "pip_prebundle")))
    if not prebundle:
        raise RuntimeError("Isaac Sim extscache 에서 omni.kit.converter.gsplat 를 찾지 못했다")
    sys.path.insert(0, prebundle[-1])
    from pxr import Gf, Usd, UsdGeom  # noqa: E402
    from usd_convert_gsplat import UP_AXIS_Z, read_ply, write_gaussian_splat_usd  # noqa: E402

    d = os.path.abspath(args.dir)
    name = args.name
    Name = name[:1].upper() + name[1:]
    crop_ply = os.path.join(d, f"{name}_crop.ply")
    splat_usd = os.path.join(d, f"{name}_splat.usd")
    scene_usda = os.path.join(d, f"{name}_scene.usda")
    xf = json.load(open(os.path.join(d, f"{name}_transform.json"), encoding="utf-8"))

    # 1) 가우시안 -> USD. 이전 실행 결과가 있으면 새로 만든다 (Stage.CreateNew 는 기존 파일에 실패할 수 있다).
    if os.path.exists(splat_usd):
        os.remove(splat_usd)
    data = read_ply(crop_ply)
    write_gaussian_splat_usd(data, splat_usd, source_file=crop_ply, prim_name=f"{Name}Splat", up_axis=UP_AXIS_Z)

    # 2) 장면: Xform(translate, orient, scale) 아래에서 1) 을 참조
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    obj = UsdGeom.Xform.Define(stage, f"/World/{Name}")
    dbl = UsdGeom.XformOp.PrecisionDouble
    obj.AddTranslateOp(precision=dbl).Set(Gf.Vec3d(*xf["translate"]))
    w, x, y, z = xf["quat_wxyz"]
    obj.AddOrientOp(precision=dbl).Set(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
    s = xf["scale"]
    obj.AddScaleOp(precision=dbl).Set(Gf.Vec3d(s, s, s))
    splat = stage.DefinePrim(f"/World/{Name}/Splat")
    splat.GetReferences().AddReference("./" + os.path.basename(splat_usd))
    stage.GetRootLayer().Export(scene_usda)

    # 확인: 참조가 풀려 가우시안 prim 이 보이는지, 월드 경계가 예상 크기인지
    check = Usd.Stage.Open(scene_usda)
    p = check.GetPrimAtPath(f"/World/{Name}/Splat")
    n = len(p.GetAttribute("positions").Get() or []) if p.GetAttribute("positions") else 0
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(p).ComputeAlignedRange()
    print(f"[build] {splat_usd} ({os.path.getsize(splat_usd) / 1e6:.1f} MB)")
    print(f"[build] {scene_usda}")
    print(f"[build] /World/{Name}/Splat type={p.GetTypeName()} positions={n:,}")
    print(f"[build] world bounds min {tuple(round(v, 3) for v in bbox.GetMin())} max {tuple(round(v, 3) for v in bbox.GetMax())}")
finally:
    app.close()
