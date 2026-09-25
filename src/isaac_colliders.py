"""
Isaac(PhysX) 장면의 충돌 물체·바닥·중력을 찾아 XPBD(원본 3DGS 좌표)에 넘길 값으로 바꾼다.

  scene = ColliderScene(stage, Rm, tw, s, exclude=["/World/Model"])   # Isaac 월드 = s·Rm·p + tw
  scene.discover()                 # CollisionAPI 가 붙은 prim 을 찾는다 (재생 시작 때 다시 부르면 새로 넣은 물체도 잡힌다)
  types, poses, dims = scene.colliders()   # 매 프레임: 지금 자세 (PhysX 강체는 PhysX 에서, 고정 물체는 USD 에서)
  plane = scene.ground(ref_point_src)      # (up, height) 원본 좌표 — 충돌 평면(Isaac Ground Plane). 없으면 None
  g_dir, g_mag = scene.gravity()           # PhysicsScene 의 중력 (원본 좌표 방향, 원본 단위/s²)

모양: Cube / Sphere / Capsule → 충돌체, Plane → 바닥. Mesh 등 나머지는 아직 지원하지 않는다 (skipped 로 알린다).
PhysX 충돌(CollisionAPI)이 없어도 bool 속성 xpbd:collider = true 인 모양은 XPBD 충돌체로 쓴다 — 로봇 링크처럼
Mesh 충돌만 있는 물체에 가우시안과 닿을 부분만 기본 모양으로 근사해 붙일 때 (robot_arm.add_xpbd_pads).
강체(RigidBodyAPI) 아래의 충돌 모양은 발견 시점의 '강체 기준 상대 자세'를 기억해 두고, 매 프레임 PhysX 가 알려 주는
강체 자세에 붙여 움직인다 (get_rigidbody_transformation, 호출당 약 2 µs). 재생 중이 아니면 USD 자세를 쓴다.
"""
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics

SHAPES = {"Cube": 1, "Sphere": 0, "Capsule": 2}


def _world(prim):
    return np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), dtype=np.float64)


def _rigid(M):
    """행벡터 규약 4×4 에서 배율을 뺀 강체 부분 (회전 + 이동)."""
    R = M[:3, :3] / np.maximum(np.linalg.norm(M[:3, :3], axis=1, keepdims=True), 1e-12)
    out = np.eye(4)
    out[:3, :3], out[3, :3] = R, M[3, :3]
    return out


def _quat_xyzw_to_rows(q):
    """(x,y,z,w) → 행벡터 규약 회전 (world_row = local_row · M 이 되도록 R 의 전치)."""
    x, y, z, w = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return R.T


class ColliderScene:
    def __init__(self, stage, Rm, tw, s, exclude=()):
        self.stage, self.Rm, self.tw, self.s = stage, np.asarray(Rm, float), np.asarray(tw, float), float(s)
        self.exclude = tuple(str(p) for p in exclude)
        self.items, self.planes, self.skipped = [], [], []
        try:
            import omni.physx
            self._px = omni.physx.get_physx_interface()
        except Exception:  # noqa: BLE001
            self._px = None

    # ── 찾기 ──
    def discover(self):
        known = {it["path"]: it for it in self.items}   # 이미 PhysX 에 있는 강체는 다시 찾아도 PhysX 자세를 바로 쓴다
        self.items, self.planes, self.skipped = [], [], []
        it = iter(Usd.PrimRange(self.stage.GetPseudoRoot()))
        for prim in it:
            path = str(prim.GetPath())
            if any(path == e or path.startswith(e + "/") for e in self.exclude):
                it.PruneChildren()
                continue
            xp = prim.GetAttribute("xpbd:collider")          # XPBD 전용 대리체 (PhysX 충돌 없음, 예: 로봇 손끝)
            xpbd_only = bool(xp and xp.Get())
            if not xpbd_only:
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                en = prim.GetAttribute("physics:collisionEnabled")
                if en and en.Get() is False:
                    continue
            tn = prim.GetTypeName()
            if tn == "Plane":
                self.planes.append(prim)
                continue
            if tn not in SHAPES:
                self.skipped.append(f"{path} ({tn})")
                continue
            body = prim
            while body and body.GetPath() != body.GetPath().absoluteRootPath and not body.HasAPI(UsdPhysics.RigidBodyAPI):
                body = body.GetParent()
            body = body if (body and body.HasAPI(UsdPhysics.RigidBodyAPI)) else None
            rel = kin = None
            if body is not None:   # 강체 기준 상대 자세 (배율 포함): M_col = rel · B(강체, 배율 없음)
                rel = _world(prim) @ np.linalg.inv(_rigid(_world(body)))
                kin = body.GetAttribute("physics:kinematicEnabled")
            self.items.append({"prim": prim, "path": path, "type": tn, "body": str(body.GetPath()) if body else None,
                               "rel": rel, "kin": kin, "fresh": known[path]["fresh"] if path in known else True})
        return self

    def summary(self):
        parts = [f"{it['path']} ({it['type']}{', rigid body' if it['body'] else ', static'})" for it in self.items]
        return (f"colliders {len(self.items)}: " + (", ".join(parts) if parts else "-")
                + f" | ground planes {len(self.planes)}" + (f" | skipped (not supported yet): {', '.join(self.skipped)}"
                                                            if self.skipped else ""))

    # ── 매 프레임 ──
    @staticmethod
    def is_dynamic(it):
        """PhysX 가 움직이는 강체에 붙은 충돌체인가 (운동학 강체·고정 물체는 아니다). 재생 중 바뀔 수 있어 매번 본다."""
        kin = it.get("kin")
        return it["body"] is not None and not (kin and kin.Get())

    def apply_reactions(self, items, push, centroid, particle_mass, dt):
        """양방향 결합: 이번 스텝에 충돌체가 가우시안에 준 충격량(= 입자 질량 × 밀어낸 변위 합 / dt)의 반대를
        그 충돌체의 PhysX 동적 강체에 접촉 중심에서 준다 (뉴턴 3법칙). items / push / centroid 는 같은 스텝·같은 순서.
        반환: 적용한 (경로, 충격량 [kg·m/s]) 목록."""
        import carb
        import omni.physx
        import omni.usd
        from pxr import PhysicsSchemaTools

        sim_if = omni.physx.get_physx_simulation_interface()
        stage_id = omni.usd.get_context().get_stage_id()
        applied = []
        for it, dp, c in zip(items, push, centroid):
            if not self.is_dynamic(it):
                continue
            J = -(particle_mass * self.s / dt) * (self.Rm @ np.asarray(dp, float))    # 월드 [kg·m/s]
            if not np.isfinite(J).all() or float(np.linalg.norm(J)) < 1e-9:
                continue
            p = self.s * (self.Rm @ np.asarray(c, float)) + self.tw
            sim_if.apply_force_at_pos(stage_id, PhysicsSchemaTools.sdfPathToInt(it["body"]),
                                      carb.Float3(*[float(v) for v in J]), carb.Float3(*[float(v) for v in p]), "Impulse")
            applied.append((it["body"], J))
        return applied

    def _current_world(self, it):
        # 운동학 강체는 USD(스크립트·뷰포트)가 자세를 정한다 — USD 가 가장 최신.
        dynamic = self.is_dynamic(it)
        fresh, it["fresh"] = it.get("fresh", False), False   # 막 찾은 강체는 PhysX 에 아직 없을 수 있다 → 첫 번은 USD
        if dynamic and not fresh and self._px is not None:
            r = self._px.get_rigidbody_transformation(it["body"])
            if r.get("ret_val"):
                B = np.eye(4)
                B[:3, :3] = _quat_xyzw_to_rows(r["rotation"])
                B[3, :3] = r["position"]
                return it["rel"] @ B
        return _world(it["prim"])

    def colliders(self):
        types, poses, dims = [], [], []
        Rt = self.Rm.T
        for it in self.items:
            M = self._current_world(it)
            A = M[:3, :3].T                      # 열벡터 규약: world = A · local + T
            T = M[3, :3]
            it["center_w"] = T.copy()            # 충돌체 중심 (월드) — 프레임 간 차이로 물체 속도를 본다
            sc = np.linalg.norm(A, axis=0)
            Rw = A / np.maximum(sc, 1e-12)
            prim, tn = it["prim"], it["type"]
            if tn == "Cube":
                d = 0.5 * float(prim.GetAttribute("size").Get() or 2.0) * sc
            elif tn == "Sphere":
                d = np.array([float(prim.GetAttribute("radius").Get() or 1.0) * sc.max(), 0.0, 0.0])
            else:   # Capsule: 축을 로컬 z 로 (순환 치환이라 오른손 좌표계 유지)
                ai = "XYZ".index(str(prim.GetAttribute("axis").Get() or "Z"))
                perm = [(ai + 1) % 3, (ai + 2) % 3, ai]
                Rw, scp = Rw[:, perm], sc[perm]
                d = np.array([float(prim.GetAttribute("radius").Get() or 0.5) * max(scp[0], scp[1]),
                              0.5 * float(prim.GetAttribute("height").Get() or 1.0) * scp[2], 0.0])
            types.append(SHAPES[tn])
            poses.append(np.concatenate([(Rt @ Rw).ravel(), Rt @ (T - self.tw) / self.s]))
            dims.append(d / self.s)
        return (np.array(types, np.int32), np.array(poses, np.float32).reshape(-1, 12),
                np.array(dims, np.float32).reshape(-1, 3))

    def ground(self, ref_point_src=None):
        """첫 충돌 평면 → (up, height) 원본 좌표 (up·p = height 가 평면). ref_point 쪽이 평면 위가 되게 방향을 맞춘다."""
        if not self.planes:
            return None
        prim = self.planes[0]
        M = _world(prim)
        ai = "XYZ".index(str(prim.GetAttribute("axis").Get() or "Z"))
        n_w = M[ai, :3] / max(np.linalg.norm(M[ai, :3]), 1e-12)     # 로컬 축 행 = 그 축의 월드 방향
        p_w = M[3, :3]
        up = self.Rm.T @ n_w
        up /= np.linalg.norm(up)
        p = self.Rm.T @ (p_w - self.tw) / self.s
        h = float(up @ p)
        if ref_point_src is not None and float(up @ np.asarray(ref_point_src, float)) < h:
            up, h = -up, -h
        return up, h

    def gravity(self, default_mag=9.81):
        """PhysicsScene 의 중력 → (원본 좌표 방향 단위벡터, 원본 단위/s²). 장면이 없으면 (−Z, default_mag)."""
        d_w, mag = np.array([0.0, 0.0, -1.0]), float(default_mag)
        for prim in self.stage.Traverse():
            if prim.IsA(UsdPhysics.Scene):
                sc = UsdPhysics.Scene(prim)
                d = sc.GetGravityDirectionAttr().Get()
                m = sc.GetGravityMagnitudeAttr().Get()
                if d is not None and np.linalg.norm(np.array(d, float)) > 1e-9:
                    d_w = np.array(d, float) / np.linalg.norm(np.array(d, float))
                if m is not None and np.isfinite(m) and m >= 0:
                    mpu = UsdGeom.GetStageMetersPerUnit(self.stage) or 1.0
                    mag = float(m) * mpu            # 장면 단위/s² → m/s² (Isaac 월드 = m)
                break
        g = self.Rm.T @ d_w
        return g / np.linalg.norm(g), mag / self.s
