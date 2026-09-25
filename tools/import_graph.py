"""
Convert a physics graph exported by the SIBR viewer into the PLY/USD order used
by the Isaac runtime.

Run with Python, NumPy, and SciPy:
  python import_graph.py --dir wolf --name wolf

Input:
  <dir>/<name>_graph_sibr.bin
  <dir>/<name>_crop.ply

Output:
  <dir>/<name>_graph.npz
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from splat_io import read_ply


def read_graph(path):
    raw = open(path, "rb").read()
    if raw[:8] != b"APGGRPH1":
        raise ValueError(f"형식이 다르다 (magic {raw[:8]!r}): {path}")
    N, M = np.frombuffer(raw, dtype="<i4", count=2, offset=8)
    N, M = int(N), int(M)
    o = 16
    def take(dtype, n):
        nonlocal o
        a = np.frombuffer(raw, dtype=dtype, count=n, offset=o)
        o += a.nbytes
        return a
    g = {
        "pos": take("<f4", N * 3).reshape(N, 3),
        "offset": take("<i4", N),
        "count": take("<i4", N),
        "idx": take("<i4", M),
        "dist": take("<f4", M),
        "stiff": take("<f4", M),
    }
    if o != len(raw):
        raise ValueError(f"파일 크기가 형식과 맞지 않는다: 읽음 {o} / 파일 {len(raw)} bytes")
    return N, M, g


def keys_of(P):
    """float32 (n,3) → 비트 패턴 키 (12바이트 void)."""
    return np.ascontiguousarray(P, dtype=np.float32).view(np.dtype((np.void, 12))).ravel()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    d = args.dir
    N, M, g = read_graph(os.path.join(d, f"{args.name}_graph_sibr.bin"))
    _, data = read_ply(os.path.join(d, f"{args.name}_crop.ply"))
    P_usd = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)
    print(f"[graph] viewer graph: {N:,} nodes, {M:,} CSR entries | USD splats: {P_usd.shape[0]:,}")

    # 비트 패턴 짝짓기
    k_usd = keys_of(P_usd)
    k_sibr = keys_of(g["pos"])
    uniq, first_usd, cnt = np.unique(k_usd, return_index=True, return_counts=True)   # 고유 키 → 첫 USD 인덱스
    dup = int((cnt > 1).sum())
    pos_in_uniq = np.searchsorted(uniq, k_sibr)
    pos_in_uniq = np.clip(pos_in_uniq, 0, len(uniq) - 1)
    found = uniq[pos_in_uniq] == k_sibr
    if dup:
        print(f"[graph] WARNING: {dup} positions appear more than once in the USD splats (ambiguous match)")
    if not found.all() or N != P_usd.shape[0]:
        miss = int((~found).sum())
        raise SystemExit(f"[graph] 짝짓기 실패: 뷰어 노드 {miss:,} 개가 USD 에 없다 (N 뷰어 {N:,} / USD {P_usd.shape[0]:,}). "
                         f"같은 PLY·크롭·opacity 조건으로 뷰어를 띄웠는지 확인할 것")
    sibr_to_usd = first_usd[pos_in_uniq].astype(np.int64)
    if len(np.unique(sibr_to_usd)) != N:
        raise SystemExit("[graph] 짝짓기가 일대일이 아니다 (중복 위치)")
    usd_to_sibr = np.empty(N, dtype=np.int64)
    usd_to_sibr[sibr_to_usd] = np.arange(N)
    print(f"[graph] matched all {N:,} nodes bit-exactly (viewer order -> USD order)")

    # CSR 재배열: USD 노드 u 의 이웃 목록 = 뷰어 노드 usd_to_sibr[u] 의 목록을 USD 인덱스로 바꾼 것
    off_s, cnt_s = g["offset"].astype(np.int64), g["count"].astype(np.int64)
    new_count = cnt_s[usd_to_sibr]
    new_offset = np.zeros(N, dtype=np.int64)
    new_offset[1:] = np.cumsum(new_count)[:-1]
    total = int(new_count.sum())
    gather = np.concatenate([np.arange(off_s[s], off_s[s] + cnt_s[s]) for s in usd_to_sibr]) if total else np.zeros(0, np.int64)
    idx_s = g["idx"][gather].astype(np.int64)
    valid = (idx_s >= 0) & (idx_s < N)
    new_idx = np.where(valid, sibr_to_usd[np.clip(idx_s, 0, N - 1)], -1).astype(np.int32)
    new_dist = g["dist"][gather].astype(np.float32)
    new_stiff = g["stiff"][gather].astype(np.float32)

    # 검증
    owner = np.repeat(np.arange(N), new_count)
    vo, vn = owner[valid], new_idx[valid].astype(np.int64)
    L = np.linalg.norm(P_usd[vo].astype(np.float64) - P_usd[vn].astype(np.float64), axis=1)
    rel = np.abs(L - new_dist[valid]) / np.maximum(L, 1e-12)
    A = coo_matrix((np.ones(len(vo)), (vo, vn)), shape=(N, N)).tocsr()
    sym = float(A.multiply(A.T).nnz) / max(A.nnz, 1)
    ncomp, labels = connected_components(A, directed=False)
    sizes = np.bincount(labels)
    spacing = float(np.median(new_dist[valid]))
    print(f"[graph] CSR entries {total:,} (invalid {int((~valid).sum())}) | degree min {int(new_count.min())} "
          f"median {int(np.median(new_count))} max {int(new_count.max())} | isolated {int((new_count == 0).sum())}")
    print(f"[graph] rest dist vs positions: max rel err {rel.max():.2e} | symmetric links {sym:.1%} | "
          f"median edge {spacing:.5f} units")
    print(f"[graph] components {ncomp} | largest {int(sizes.max()):,} ({sizes.max() / N:.4%}) | "
          f"stiffness [{new_stiff.min():.3f}, {new_stiff.max():.3f}] mean {new_stiff.mean():.3f}")

    out = os.path.join(d, f"{args.name}_graph.npz")
    np.savez_compressed(out, offset=new_offset.astype(np.int32), count=new_count.astype(np.int32), idx=new_idx,
                        dist=new_dist, stiff=new_stiff, pos_rest=P_usd, sibr_to_usd=sibr_to_usd.astype(np.int32))
    print(f"[graph] wrote {out}")


if __name__ == "__main__":
    main()
