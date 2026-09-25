"""
Crop a 3D Gaussian Splatting PLY and calculate the OpenUSD placement transform.

Run with Python and NumPy:
  python prepare_splat.py --ply <point_cloud.ply> --crop-json <case.json> --cameras <cameras.json> --out-dir wolf --name wolf

Output:
  <out-dir>/<name>_crop.ply
  <out-dir>/<name>_transform.json
"""
import argparse
import json
import math
import os

import numpy as np


def read_ply(path):
    """binary_little_endian, vertex 원소 하나, 전부 float 속성인 3DGS PLY 를 구조체 배열로 읽는다."""
    header = []
    with open(path, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"PLY header 가 끝나지 않았다: {path}")
            s = line.decode("ascii").rstrip("\r\n")
            header.append(s)
            if s == "end_header":
                break
        offset = f.tell()
    if "format binary_little_endian 1.0" not in header:
        raise ValueError("binary_little_endian PLY 만 지원한다")
    count = None
    names = []
    for s in header:
        tok = s.split()
        if not tok:
            continue
        if tok[0] == "element":
            if tok[1] != "vertex" or count is not None:
                raise ValueError(f"vertex 원소 하나만 지원한다: {s}")
            count = int(tok[2])
        elif tok[0] == "property":
            if tok[1] != "float":
                raise ValueError(f"float 이 아닌 속성: {s}")
            names.append(tok[2])
    data = np.fromfile(path, dtype=np.dtype([(n, "<f4") for n in names]), count=count, offset=offset)
    if data.shape[0] != count:
        raise ValueError(f"vertex {count} 개를 기대했는데 {data.shape[0]} 개만 읽혔다")
    return header, data


def write_ply(path, header, data):
    out = [f"element vertex {data.shape[0]}" if s.startswith("element vertex") else s for s in header]
    with open(path, "wb") as f:
        f.write(("\n".join(out) + "\n").encode("ascii"))
        f.write(data.tobytes())


def rotation_between(a, b):
    """단위벡터 a 를 b 로 보내는 최소 회전 행렬."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -0.999999:   # 정반대: a 에 수직인 축으로 180도
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


def quat_wxyz(R):
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0.0:
        s = 2.0 * math.sqrt(t + 1.0)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return (q / np.linalg.norm(q)).tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ply", required=True)
    ap.add_argument("--crop-json", help="crop_min / crop_max 가 든 실험 설정 JSON (ablation/<scene>/case_*.json)")
    ap.add_argument("--crop-min", type=float, nargs=3)
    ap.add_argument("--crop-max", type=float, nargs=3)
    ap.add_argument("--cameras", required=True, help="3DGS 출력의 cameras.json (rotation = camera-to-world)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--target-height", type=float, default=0.3, help="Isaac Sim 에서의 물체 높이 [m]")
    ap.add_argument("--bound-pct", type=float, default=0.2, help="높이·바닥을 정할 때 양 끝에서 버릴 백분위 (floater 영향 차단)")
    ap.add_argument("--min-opacity", type=float, default=0.0899, help="뷰어 loadPly 와 같은 활성화 opacity 하한")
    ap.add_argument("--yaw-deg", type=float, default=0.0,
                    help="자동 방향 뒤에 +Z 축으로 더 돌린다 (카메라가 물체를 둘러싸서 정면을 못 정할 때 손으로 정면을 +X 로)")
    args = ap.parse_args()

    if args.crop_json:
        cfg = json.load(open(args.crop_json, encoding="utf-8"))
        lo, hi = np.array(cfg["crop_min"], float), np.array(cfg["crop_max"], float)
    elif args.crop_min and args.crop_max:
        lo, hi = np.array(args.crop_min, float), np.array(args.crop_max, float)
    else:
        lo = hi = None

    header, data = read_ply(args.ply)
    P = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)
    alpha = 1.0 / (1.0 + np.exp(-data["opacity"].astype(np.float64)))
    keep = alpha >= args.min_opacity
    if lo is not None:
        keep &= np.all((P >= lo) & (P <= hi), axis=1)
    crop = data[keep]
    Pc = P[keep]
    print(f"[prepare] {os.path.basename(args.ply)}: {data.shape[0]:,} -> {crop.shape[0]:,} splats "
          f"(box {lo.tolist() if lo is not None else 'none'} .. {hi.tolist() if hi is not None else 'none'}, "
          f"opacity >= {args.min_opacity})")

    # 카메라: 3DGS cameras.json 의 rotation 은 camera-to-world. COLMAP 카메라 y 축이 아래를 향하므로
    # 월드 up = −(2번째 열), 카메라가 보는 방향 = 3번째 열.
    cams = json.load(open(args.cameras, encoding="utf-8"))
    Rc = np.array([c["rotation"] for c in cams], dtype=np.float64)
    ups = -Rc[:, :, 1]
    fwd = Rc[:, :, 2]
    up_sum = ups.sum(axis=0)
    up = up_sum / np.linalg.norm(up_sum)
    up_consistency = np.linalg.norm(up_sum) / len(cams)          # 1 = 모든 카메라 up 이 같음
    fwd_mean = fwd.mean(axis=0)
    fwd_consistency = float(np.linalg.norm(fwd_mean))           # 0 에 가까우면 360도 촬영 → 정면 없음
    print(f"[prepare] cameras {len(cams)} | up {np.round(up, 4).tolist()} (consistency {up_consistency:.3f}) "
          f"| mean view dir norm {fwd_consistency:.3f}")

    R = rotation_between(up, np.array([0.0, 0.0, 1.0]))
    yaw_note = "skipped (cameras surround the object)"
    if fwd_consistency > 0.3:
        front = R @ (-fwd_mean / fwd_consistency)               # 물체 정면 = 카메라들이 보는 방향의 반대
        fh = np.array([front[0], front[1], 0.0])
        if np.linalg.norm(fh) > 0.2:
            a = -math.atan2(fh[1], fh[0])                       # 정면을 +X 로
            Rz = np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])
            R = Rz @ R
            yaw_note = "front -> +X"
    if args.yaw_deg:
        a = math.radians(args.yaw_deg)
        R = np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]]) @ R
        yaw_note += f", then yaw {args.yaw_deg:+g} deg"

    Q = Pc @ R.T
    p = args.bound_pct
    qlo = np.percentile(Q, p, axis=0)
    qhi = np.percentile(Q, 100.0 - p, axis=0)
    height = float(qhi[2] - qlo[2])
    s = args.target_height / height
    cx, cy = 0.5 * (qlo[0] + qhi[0]), 0.5 * (qlo[1] + qhi[1])
    t = [-s * cx, -s * cy, -s * float(qlo[2])]                # 바닥(p 백분위) z = 0, 수평 중심 = 원점

    os.makedirs(args.out_dir, exist_ok=True)
    ply_out = os.path.join(args.out_dir, f"{args.name}_crop.ply")
    write_ply(ply_out, header, crop)

    raw_h = float(Q[:, 2].max() - Q[:, 2].min())
    xf = {
        "source_ply": os.path.abspath(args.ply),
        "crop_min": lo.tolist() if lo is not None else None,
        "crop_max": hi.tolist() if hi is not None else None,
        "splats": int(crop.shape[0]),
        "up_source": up.tolist(),
        "up_consistency": up_consistency,
        "yaw": yaw_note,
        "rotation_matrix": R.tolist(),
        "quat_wxyz": quat_wxyz(R),
        "scale": s,
        "translate": t,
        "height_units_robust": height,
        "height_units_raw": raw_h,
        "target_height_m": args.target_height,
        "bound_pct": p,
        "note": "USD xformOpOrder = translate, orient, scale  ->  world = s * R * p + t",
    }
    xf_out = os.path.join(args.out_dir, f"{args.name}_transform.json")
    with open(xf_out, "w", encoding="utf-8") as f:
        json.dump(xf, f, indent=2)

    ext = (qhi - qlo) * s
    print(f"[prepare] orientation: up -> +Z, {yaw_note}")
    print(f"[prepare] height {height:.4f} units (raw {raw_h:.4f}) -> {args.target_height:.3f} m, scale {s:.6f}")
    print(f"[prepare] size in Isaac Sim (m): x {ext[0]:.3f}  y {ext[1]:.3f}  z {ext[2]:.3f}")
    print(f"[prepare] wrote {ply_out} ({os.path.getsize(ply_out) / 1e6:.1f} MB)")
    print(f"[prepare] wrote {xf_out}")


if __name__ == "__main__":
    main()
