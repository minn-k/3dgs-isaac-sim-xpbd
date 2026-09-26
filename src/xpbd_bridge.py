"""ctypes bridge between the Python runtime and the CUDA XPBD solver.

The solver receives 3DGS positions, graph constraints, gripper attachments,
and collision proxies. It returns deformed Gaussian positions, orientations,
and scales in the same PLY/USD ordering.
"""
import ctypes
import os

import numpy as np

from splat_io import read_ply

_HERE = os.path.dirname(os.path.abspath(__file__))


def load_inputs(d, name):
    """<d>/<name>_crop.ply 와 <d>/<name>_graph.npz 로 DLL 입력을 만든다 (둘 다 USD 순서)."""
    _, data = read_ply(os.path.join(d, f"{name}_crop.ply"))
    g = np.load(os.path.join(d, f"{name}_graph.npz"))
    pos = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)
    if not np.array_equal(pos, g["pos_rest"]):
        raise ValueError("그래프(pos_rest)와 크롭 PLY 의 위치가 다르다 — import_graph.py 를 다시 돌릴 것")
    scales = np.exp(np.stack([data["scale_0"], data["scale_1"], data["scale_2"]], axis=1)).astype(np.float32)
    rots = np.stack([data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"]], axis=1).astype(np.float32)
    rots /= np.linalg.norm(rots, axis=1, keepdims=True)
    opacity = (1.0 / (1.0 + np.exp(-data["opacity"].astype(np.float64)))).astype(np.float32)
    # CSR(양방향) → 무향 간선 i<j 한 번씩 (SIBR loadGraph 입력과 같은 형태)
    owner = np.repeat(np.arange(len(pos), dtype=np.int32), g["count"])
    idx = g["idx"]
    keep = idx > owner
    edges = np.stack([owner[keep], idx[keep]], axis=1).astype(np.int32)
    return {
        "pos": np.ascontiguousarray(pos),
        "scales": np.ascontiguousarray(scales),
        "rots": np.ascontiguousarray(rots),
        "opacity": np.ascontiguousarray(opacity),
        "edges": np.ascontiguousarray(edges),
        "rest": np.ascontiguousarray(g["dist"][keep].astype(np.float32)),
        "stiff": np.ascontiguousarray(g["stiff"][keep].astype(np.float32)),
    }



class XPBD:
    def __init__(self, dll_path=None):
        default_dll = os.path.join(os.path.dirname(_HERE), "bin", "xpbd_isaac.dll")
        path = dll_path or os.environ.get("XPBD_DLL") or default_dll
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"XPBD DLL not found: {path}. Build native/ or place the matching release asset in bin/."
            )
        self.lib = ctypes.CDLL(path)
        L = self.lib
        f32 = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")
        i32 = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
        c_int, c_float = ctypes.c_int, ctypes.c_float
        L.xpbd_last_error.restype = ctypes.c_char_p
        L.xpbd_create.argtypes = [c_int, f32, f32, f32, f32, c_int, i32, f32, f32]
        L.xpbd_create.restype = c_int
        L.xpbd_destroy.restype = None
        L.xpbd_count.restype = c_int
        L.xpbd_set_solver.argtypes = [c_int] + [c_float] * 10
        L.xpbd_set_solver.restype = None
        L.xpbd_set_constraints.argtypes = [c_int] * 4
        L.xpbd_set_constraints.restype = None
        L.xpbd_set_volume.argtypes = [c_float, c_int, c_int, c_int]
        L.xpbd_set_volume.restype = None
        L.xpbd_set_ground.argtypes = [c_int] + [c_float] * 9
        L.xpbd_set_ground.restype = None
        L.xpbd_set_object_shape.argtypes = [c_float]
        L.xpbd_set_object_shape.restype = None
        L.xpbd_set_shape_robust.argtypes = [c_int]
        L.xpbd_set_shape_robust.restype = None
        L.xpbd_set_object_shape_gpu.argtypes = [c_int]
        L.xpbd_set_object_shape_gpu.restype = None
        L.xpbd_set_self_collision.argtypes = [c_int, c_float, c_float, c_int]
        L.xpbd_set_self_collision.restype = None
        L.xpbd_self_collision_stats.argtypes = [i32]
        L.xpbd_self_collision_stats.restype = None
        L.xpbd_set_colliders.argtypes = [c_int, i32, f32, f32, c_float, c_float]
        L.xpbd_set_colliders.restype = None
        L.xpbd_collider_stats.argtypes = [i32, f32, f32, c_int]
        L.xpbd_collider_stats.restype = c_int
        L.xpbd_add_rigid_velocity.argtypes = [f32, f32, f32]
        L.xpbd_add_rigid_velocity.restype = c_int
        self.has_attach = hasattr(L, "xpbd_set_attached")      # 예전 DLL 에는 없다
        if self.has_attach:
            L.xpbd_set_attached.argtypes = [c_int, i32, f32, c_float]
            L.xpbd_set_attached.restype = c_int
        self.has_contact_shape = hasattr(L, "xpbd_set_contact_shape")   # 2026-09-25 빌드부터
        if self.has_contact_shape:
            L.xpbd_set_contact_shape.argtypes = [c_int, c_float, c_float]
            L.xpbd_set_contact_shape.restype = c_int
        L.xpbd_step.restype = c_int
        L.xpbd_positions_device.restype = ctypes.c_ulonglong
        L.xpbd_positions_host.argtypes = [f32, c_int]
        L.xpbd_positions_host.restype = c_int
        L.xpbd_reset.restype = c_int
        L.xpbd_launch.argtypes = [f32, f32, f32]
        L.xpbd_launch.restype = c_int
        L.xpbd_volume_stats.argtypes = [c_int, ctypes.POINTER(c_float), ctypes.POINTER(c_float)]
        L.xpbd_volume_stats.restype = None
        L.xpbd_compute_shapes.argtypes = [c_float, f32, f32, c_int]
        L.xpbd_compute_shapes.restype = c_int
        L.xpbd_press_start.argtypes = [c_int, c_float, c_float, c_int]
        L.xpbd_press_start.restype = c_int
        L.xpbd_press_stop.restype = None
        L.xpbd_press_autolog.argtypes = [c_int]
        L.xpbd_press_autolog.restype = None
        L.xpbd_press_state.argtypes = [ctypes.POINTER(c_float)] * 4
        L.xpbd_press_state.restype = c_int
        self.n = 0

    def _check(self, rc, what):
        if rc != 0:
            raise RuntimeError(f"{what}: {self.lib.xpbd_last_error().decode(errors='replace')}")

    def create(self, inp):
        e = inp["edges"].shape[0]
        self._check(self.lib.xpbd_create(inp["pos"].shape[0], inp["pos"].ravel(), inp["scales"].ravel(),
                                         inp["rots"].ravel(), inp["opacity"], e, inp["edges"].ravel(),
                                         inp["rest"], inp["stiff"]), "xpbd_create")
        self.n = inp["pos"].shape[0]

    def set_solver(self, iters=20, dt=1 / 60, under_relax=0.6, vel_damping=0.01, stiffness_scale=1.0,
                   inv_mass_scale=1.0, dist_compliance=0.0, shape_compliance=0.0, shape_blend=0.1,
                   angle_compliance=1e-2, angle_blend=0.2):
        self.lib.xpbd_set_solver(iters, dt, under_relax, vel_damping, stiffness_scale, inv_mass_scale,
                                 dist_compliance, shape_compliance, shape_blend, angle_compliance, angle_blend)

    def set_constraints(self, distance=True, shape=True, angle=False, volume=True):
        self.lib.xpbd_set_constraints(int(distance), int(shape), int(angle), int(volume))

    def set_volume(self, compliance=1e-6, ring_k=3, max_members=2048, leader_min_hop=2):
        self.lib.xpbd_set_volume(compliance, ring_k, max_members, leader_min_hop)

    def set_ground(self, enabled, up, height, friction=0.5, restitution=0.0, contact_radius=0.0,
                   contact_slop=0.0, gravity=0.0):
        self.lib.xpbd_set_ground(int(enabled), float(up[0]), float(up[1]), float(up[2]), height, friction,
                                 restitution, contact_radius, contact_slop, gravity)

    def set_object_shape(self, stiffness):
        self.lib.xpbd_set_object_shape(stiffness)

    def set_object_shape_gpu(self, enabled):
        """물체 단위 형상 유지의 누적·극분해. True = GPU (기본값), False = 기존 호스트 경로 (A/B 비교용)."""
        self.lib.xpbd_set_object_shape_gpu(int(enabled))

    SPHERE, BOX, CAPSULE = 0, 1, 2

    def set_colliders(self, types, poses, dims, margin=0.0, friction=0.0):
        """운동학 충돌체 (한 방향: 충돌체 → 가우시안). 입력 좌표계(원본 3DGS) 값. 매 스텝 전에 다시 불러도 된다.
        types (K,) int: 0 sphere / 1 box / 2 capsule
        poses (K,12): 로컬→입력 회전 R (행우선 9) + 중심 t (3)
        dims  (K,3) : box 반 크기 / sphere (반지름,0,0) / capsule (반지름, 반 길이(로컬 z), 0)
        빈 목록이면 충돌체를 모두 끈다."""
        types = np.ascontiguousarray(np.asarray(types, dtype=np.int32).reshape(-1))
        k = len(types)
        poses = np.ascontiguousarray(np.asarray(poses, dtype=np.float32).reshape(k, 12) if k else np.zeros((1, 12), np.float32))
        dims = np.ascontiguousarray(np.asarray(dims, dtype=np.float32).reshape(k, 3) if k else np.zeros((1, 3), np.float32))
        if not k:
            types = np.zeros(1, dtype=np.int32)
        self.lib.xpbd_set_colliders(k, types, poses.ravel(), dims.ravel(), float(margin), float(friction))

    def collider_stats(self, count):
        """마지막 스텝의 충돌체별 (닿은 입자 수 (K,), 밀어낸 변위 합 (K,3), 접촉 중심 (K,3)) — 입력 좌표·단위.
        밀어낸 변위 합 × 입자 질량 / dt = 이번 스텝에 충돌체가 가우시안에 준 충격량 (반작용은 그 반대)."""
        hits = np.zeros(max(count, 1), dtype=np.int32)
        push = np.zeros(max(count, 1) * 3, dtype=np.float32)
        cen = np.zeros(max(count, 1) * 3, dtype=np.float32)
        n = self.lib.xpbd_collider_stats(hits, push, cen, count)
        return hits[:n], push[:3 * n].reshape(n, 3), cen[:3 * n].reshape(n, 3)

    def add_rigid_velocity(self, dv, dw=(0.0, 0.0, 0.0), center=(0.0, 0.0, 0.0)):
        """모든 가우시안 속도에 dv + dw × (x − center) 를 더한다 (입력 좌표계·단위/s, 스텝 사이에 부른다)."""
        a = [np.ascontiguousarray(v, dtype=np.float32) for v in (dv, dw, center)]
        self._check(self.lib.xpbd_add_rigid_velocity(*a), "xpbd_add_rigid_velocity")

    def set_attached(self, idx, pos, weight=1.0):
        """붙잡힌 가우시안 (로봇 손가락 등): idx 를 고정하고 위치를 pos (N×3, 입력 좌표)로. 스텝마다 새 위치로 부른다.
        weight = 물체 단위 형상 유지의 강체 맞춤에서 붙잡힌 점 하나의 무게 (몸 전체가 붙잡힌 곳을 따라오게)."""
        if not self.has_attach:
            raise RuntimeError("This XPBD DLL does not export xpbd_set_attached. Rebuild native/.")
        idx = np.ascontiguousarray(idx, dtype=np.int32).ravel()
        pos = np.ascontiguousarray(pos, dtype=np.float32).reshape(-1, 3)
        self._check(self.lib.xpbd_set_attached(len(idx), idx, pos.ravel(), float(weight)), "xpbd_set_attached")

    def clear_attached(self):
        if self.has_attach:
            self._check(self.lib.xpbd_set_attached(0, np.zeros(1, np.int32), np.zeros(3, np.float32), 1.0),
                        "xpbd_set_attached")

    def set_contact_shape(self, enabled, tau=0.2, radius_cap=0.0):
        """타원체 접촉: 충돌체·바닥이 가우시안 중심이 아니라 불투명도 tau 등고면 타원체와 부딪힌다.
        enabled: 1 = 충돌체(로봇 손가락·던진 물체)만, 2 = 바닥만, 3 = 둘 다, 0/False = 끔 (True = 1).
        radius_cap = 반축 상한 (입력 단위, 0 = 없음). 켜면 shapes() 가 부를 때마다 변형된 모양으로 다시 묶는다.
        예전 DLL 이면 False 를 돌려준다 (중심점 접촉 그대로)."""
        if not self.has_contact_shape:
            return False
        self._check(self.lib.xpbd_set_contact_shape(int(enabled), float(tau), float(radius_cap)),
                    "xpbd_set_contact_shape")
        return True

    def set_self_collision(self, enabled, radius_scale=1.5, exclude_scale=2.0, within_body=False):
        self.lib.xpbd_set_self_collision(int(enabled), radius_scale, exclude_scale, int(within_body))

    def self_collision_stats(self):
        """(후보 수, 활성 입자 수, 상한 초과 교체 수) — 마지막 스텝."""
        out = np.zeros(3, dtype=np.int32)
        self.lib.xpbd_self_collision_stats(out)
        return int(out[0]), int(out[1]), int(out[2])

    def set_shape_robust(self, enabled):
        """형상 제약 회전 추출. True = double 극분해 (SIBR "robust rotation", 느림), False = float32 경로."""
        self.lib.xpbd_set_shape_robust(int(enabled))

    def step(self):
        self._check(self.lib.xpbd_step(), "xpbd_step")

    def positions(self):
        out = np.empty((self.n, 3), dtype=np.float32)
        self._check(self.lib.xpbd_positions_host(out.ravel(), self.n), "xpbd_positions_host")
        return out

    def shapes(self, deform_eps=1e-2):
        """변형된 가우시안 모양 (SIBR 렌더 경로와 같은 규칙).
        반환 (scales (N,3) activated, quats (N,4) w,x,y,z, 변형 반영 수). deform_eps 이하로 움직인 가우시안은 원래 값."""
        sc = np.empty((self.n, 3), dtype=np.float32)
        q = np.empty((self.n, 4), dtype=np.float32)
        k = self.lib.xpbd_compute_shapes(float(deform_eps), sc.ravel(), q.ravel(), self.n)
        if k < 0:
            self._check(-1, "xpbd_compute_shapes")
        return sc, q, k

    def positions_device_ptr(self):
        return int(self.lib.xpbd_positions_device())

    def reset(self):
        self._check(self.lib.xpbd_reset(), "xpbd_reset")

    def launch(self, lin=(0, 0, 0), ang=(0, 0, 0), tilt=(0, 0, 0)):
        a = [np.ascontiguousarray(v, dtype=np.float32) for v in (lin, ang, tilt)]
        self._check(self.lib.xpbd_launch(*a), "xpbd_launch")

    def volume_stats(self, enable=True):
        m, s = ctypes.c_float(0.0), ctypes.c_float(0.0)
        self.lib.xpbd_volume_stats(int(enable), ctypes.byref(m), ctypes.byref(s))
        return m.value, s.value

    def press_start(self, axis, ramp_per_sec, max_disp_pct, from_lo):
        """축(0/1/2) 방향 평판 프레스. from_lo=True 면 min 쪽 평판이 움직인다. ramp 는 입력 좌표 단위/초."""
        self._check(self.lib.xpbd_press_start(int(axis), float(ramp_per_sec), float(max_disp_pct), int(from_lo)),
                    "xpbd_press_start")

    def press_autolog(self, enabled):
        """25/50/75/100% 체크포인트 정착(멈춤) on/off. 다음 press_start 부터. 기본 on."""
        self.lib.xpbd_press_autolog(int(enabled))

    def press_stop(self):
        self.lib.xpbd_press_stop()

    def press_state(self):
        """(active, lo, hi, cur_disp, max_disp) — lo/hi 는 현재 평판의 축 좌표."""
        v = [ctypes.c_float(0.0) for _ in range(4)]
        active = self.lib.xpbd_press_state(*[ctypes.byref(x) for x in v])
        return bool(active), v[0].value, v[1].value, v[2].value, v[3].value

    def destroy(self):
        self.lib.xpbd_destroy()
        self.n = 0
