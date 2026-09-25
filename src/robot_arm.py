"""
Isaac Franka Panda 를 스크립트로 부리는 도구: 순기구학(FK)·역기구학(IK)·관절 목표 쓰기·XPBD 접촉 대리체.

  arm = FrankaArm(stage, "/World/RobotBase/Franka")      # USD 관절 정의에서 사슬을 읽는다 (재생 전)
  q = arm.ik(p_target, R_target, q_start)                # 손끝(TCP) 월드 위치·자세 → 팔 관절 7개 [rad]
  arm.set_targets(q, finger=0.0)                         # PhysX 관절 드라이브 목표 (USD drive:targetPosition)
  arm.add_xpbd_pads()                                    # 손가락·손바닥에 캡슐·상자 대리체 (XPBD 만 본다)

FK 는 USD Physics 관절 규약 그대로: body1 = body0 · F0 · D(q) · F1⁻¹ (F = localPos·localRot, D = 관절축 회전/이동).
관절 축이 모두 로컬 X 라 D(q) 는 X 축 회전(회전 관절)·X 축 이동(손가락). 행렬은 열벡터 규약 4×4.
로봇이 실제로 어디 있는지는 PhysX 가 정한다 — 여기 FK 는 목표를 정할 때만 쓰고, 접촉은 PhysX 링크 자세를 따른다.
"""
import numpy as np

URL = ("https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1/Isaac/"
       "Robots/FrankaRobotics/FrankaPanda/franka.usd")
ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
FINGER_JOINTS = ["panda_finger_joint1", "panda_finger_joint2"]
TCP_OFFSET = 0.1034          # panda_hand 원점 → 손가락 끝 가운데 [m] (hand z 축)
HOME_DEG = (0.0, -45.0, 0.0, -135.0, 0.0, 90.0, 45.0)


def quat_wxyz_to_mat(q):
    w, x, y, z = (float(v) for v in q)
    n = np.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _frame(pos, rot_wxyz):
    F = np.eye(4)
    F[:3, :3] = quat_wxyz_to_mat(rot_wxyz)
    F[:3, 3] = pos
    return F


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    D = np.eye(4)
    D[1, 1], D[1, 2], D[2, 1], D[2, 2] = c, -s, s, c
    return D


def _trans_x(d):
    D = np.eye(4)
    D[0, 3] = d
    return D


def _gfquat_wxyz(q):
    return (q.GetReal(), *q.GetImaginary())


def rot_err(R, Rt):
    """자세 오차 (작을 때 회전 벡터): 0.5 Σ r_i × r*_i."""
    return 0.5 * (np.cross(R[:, 0], Rt[:, 0]) + np.cross(R[:, 1], Rt[:, 1]) + np.cross(R[:, 2], Rt[:, 2]))


def mat_to_quat_wxyz(R):
    """회전 행렬 → 쿼터니언 (w, x, y, z)."""
    t = float(np.trace(R))
    if t > 0:
        k = 0.5 / np.sqrt(t + 1.0)
        return 0.25 / k, (R[2, 1] - R[1, 2]) * k, (R[0, 2] - R[2, 0]) * k, (R[1, 0] - R[0, 1]) * k
    i = int(np.argmax(np.diag(R)))
    j, m = (i + 1) % 3, (i + 2) % 3
    k = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[m, m])
    q = [0.0, 0.0, 0.0, 0.0]
    q[0] = (R[m, j] - R[j, m]) / k
    q[1 + i] = 0.25 * k
    q[1 + j] = (R[j, i] + R[i, j]) / k
    q[1 + m] = (R[m, i] + R[i, m]) / k
    return tuple(q)


def approach_frame(approach, side):
    """손 자세: z = 다가가는 방향(표면 안쪽), y = 손가락이 벌어지는 방향(side 를 z 에 수직으로 투영)."""
    z = np.asarray(approach, float)
    z = z / np.linalg.norm(z)
    y = np.asarray(side, float) - z * float(np.dot(side, z))
    if np.linalg.norm(y) < 1e-6:
        y = np.cross(z, [1.0, 0.0, 0.0])
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    return np.column_stack([x, y, z])


class FrankaArm:
    def __init__(self, stage, root_path):
        from pxr import Usd, UsdGeom, UsdPhysics

        self.stage, self.root = stage, root_path
        self.joints = {}
        for p in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
            if p.IsA(UsdPhysics.Joint):
                b0 = p.GetRelationship("physics:body0").GetTargets()
                b1 = p.GetRelationship("physics:body1").GetTargets()
                lim = (p.GetAttribute("physics:lowerLimit").Get(), p.GetAttribute("physics:upperLimit").Get()) \
                    if p.GetAttribute("physics:lowerLimit") else (None, None)
                self.joints[p.GetName()] = {
                    "prim": p, "type": p.GetTypeName(),
                    "body0": str(b0[0]) if b0 else None, "body1": str(b1[0]) if b1 else None,
                    "F0": _frame(p.GetAttribute("physics:localPos0").Get() or (0, 0, 0),
                                 _gfquat_wxyz(p.GetAttribute("physics:localRot0").Get())),
                    "F1inv": np.linalg.inv(_frame(p.GetAttribute("physics:localPos1").Get() or (0, 0, 0),
                                                  _gfquat_wxyz(p.GetAttribute("physics:localRot1").Get()))),
                    "lim": lim,
                }
        missing = [j for j in ARM_JOINTS + ["panda_hand_joint"] + FINGER_JOINTS if j not in self.joints]
        if missing:
            raise RuntimeError(f"Franka joints not found under {root_path}: {missing}")
        link0 = stage.GetPrimAtPath(f"{root_path}/panda_link0")
        self.base = np.array(UsdGeom.Xformable(link0).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),
                             dtype=np.float64).T              # USD 행벡터 규약 → 열벡터
        self.lo = np.radians([self.joints[j]["lim"][0] for j in ARM_JOINTS])
        self.hi = np.radians([self.joints[j]["lim"][1] for j in ARM_JOINTS])
        self.home = np.radians(HOME_DEG)
        self.finger_cmd = 0.0                              # 마지막으로 보낸 손가락 벌림 [m]

    # ── 순기구학 ──
    def fk(self, q, finger=0.0, axes=None):
        """관절 7개 [rad] (+ 손가락 벌림 [m]) → 링크 이름: 월드 4×4 (열벡터). axes 리스트를 주면 관절마다
        (월드 회전축, 월드 원점) 을 채운다 (기하학적 야코비안용)."""
        T = {"panda_link0": self.base}
        prev = "panda_link0"
        for i, jn in enumerate(ARM_JOINTS):
            J = self.joints[jn]
            nxt = J["body1"].rsplit("/", 1)[-1]
            Wj = T[prev] @ J["F0"]                       # 관절 프레임 (회전 전) — 축 = 그 X 축
            if axes is not None:
                axes.append((Wj[:3, 0].copy(), Wj[:3, 3].copy()))
            T[nxt] = Wj @ _rot_x(q[i]) @ J["F1inv"]
            prev = nxt
        J = self.joints["panda_hand_joint"]
        T["panda_hand"] = T["panda_link7"] @ J["F0"] @ J["F1inv"]
        for jn in FINGER_JOINTS:
            J = self.joints[jn]
            T[J["body1"].rsplit("/", 1)[-1]] = T["panda_hand"] @ J["F0"] @ _trans_x(finger) @ J["F1inv"]
        return T

    def tcp(self, q):
        """손가락 끝 가운데 (월드 위치, 손 자세 3×3). 손 z 축 = 다가가는 방향."""
        H = self.fk(q)["panda_hand"]
        return H[:3, 3] + TCP_OFFSET * H[:3, 2], H[:3, :3]

    # ── 역기구학 (감쇠 최소제곱 + 여유 자유도는 기본 자세 쪽으로) ──
    def ik(self, p_t, R_t, q0, iters=60, tol=2e-4, lam=0.03, w_rot=0.3, k_null=0.05):
        q = np.clip(np.array(q0, float), self.lo, self.hi)
        for _ in range(iters):
            ax = []
            H = self.fk(q, axes=ax)["panda_hand"]
            p, R = H[:3, 3] + TCP_OFFSET * H[:3, 2], H[:3, :3]
            e = np.concatenate([p_t - p, w_rot * rot_err(R, R_t)])
            if np.linalg.norm(e[:3]) < tol and np.linalg.norm(e[3:]) < tol * 5:
                break
            Jm = np.zeros((6, 7))                       # 회전 관절 i: 선속도 a×(p−o), 각속도 a
            for i, (a, o) in enumerate(ax):
                Jm[:3, i] = np.cross(a, p - o)
                Jm[3:, i] = w_rot * a
            JJt = Jm @ Jm.T + (lam ** 2) * np.eye(6)
            Jpinv = Jm.T @ np.linalg.inv(JJt)
            step = Jpinv @ e + (np.eye(7) - Jpinv @ Jm) @ (k_null * (self.home - q))
            n = np.linalg.norm(step)
            if n > 0.2:
                step *= 0.2 / n
            q = np.clip(q + step, self.lo, self.hi)
        p, R = self.tcp(q)
        return q, float(np.linalg.norm(p_t - p)), float(np.linalg.norm(rot_err(R, R_t)))

    # ── PhysX 드라이브 (USD) ──
    def set_targets(self, q, finger=None):
        for i, jn in enumerate(ARM_JOINTS):
            self.joints[jn]["prim"].GetAttribute("drive:angular:physics:targetPosition").Set(float(np.degrees(q[i])))
        if finger is not None:
            self.finger_cmd = float(finger)
            for jn in FINGER_JOINTS:        # 두 번째 손가락은 드라이브 없이 첫 번째를 따라간다 (mimic)
                a = self.joints[jn]["prim"].GetAttribute("drive:linear:physics:targetPosition")
                if a and a.GetTypeName():
                    a.Set(float(finger))

    def set_state(self, q, finger=0.0):
        """재생 시작 자세 (JointStateAPI). 목표도 같이 맞춰 첫 스텝에 팔이 튀지 않게 한다."""
        from pxr import PhysxSchema

        def _state(p, kind, value):
            st = PhysxSchema.JointStateAPI.Apply(p, kind)        # 이미 있으면 그대로
            st.CreatePositionAttr().Set(float(value))
            st.CreateVelocityAttr().Set(0.0)

        for i, jn in enumerate(ARM_JOINTS):
            _state(self.joints[jn]["prim"], "angular", np.degrees(q[i]))
        for jn in FINGER_JOINTS:
            _state(self.joints[jn]["prim"], "linear", finger)
        self.set_targets(q, finger)

    def finger_state(self):
        """PhysX 가 USD 에 써 준 지금 손가락 벌림 [m] (재생 중)."""
        a = self.joints[FINGER_JOINTS[0]]["prim"].GetAttribute("state:linear:physics:position")
        v = a.Get() if a else None
        return float(v) if v is not None else self.finger_cmd

    def joint_state(self):
        """PhysX 가 USD 에 써 준 지금 관절 각 [rad] (재생 중)."""
        return np.radians([self.joints[jn]["prim"].GetAttribute("state:angular:physics:position").Get() or 0.0
                           for jn in ARM_JOINTS])

    # ── XPBD 접촉 대리체 ──
    def add_xpbd_pads(self, visible=False, pad_radius=0.0095):
        """손가락 끝(캡슐)·손바닥(상자)에 XPBD 전용 충돌 모양을 붙인다. PhysX 충돌은 없다 (CollisionAPI 없음,
        xpbd:collider 표시만) — 로봇 Mesh 충돌체는 XPBD 가 못 읽어서, 가우시안과 닿는 부분만 기본 모양으로 근사한다.
        링크의 자식이라 isaac_colliders 가 매 스텝 그 링크의 PhysX 자세를 따라 움직인다."""
        from pxr import Gf, Sdf, UsdGeom

        made = []
        for fin, sgn in (("panda_leftfinger", 1.0), ("panda_rightfinger", -1.0)):
            cap = UsdGeom.Capsule.Define(self.stage, f"{self.root}/{fin}/xpbd_pad")
            cap.CreateAxisAttr("Z")                            # 손가락 링크 z = 손 z = 다가가는 방향
            cap.CreateRadiusAttr(pad_radius)
            cap.CreateHeightAttr(0.020)
            xf = UsdGeom.Xformable(cap)
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(0.0, sgn * 0.0085, 0.026))   # 손가락 패드 가운데 (끝 = z 0.0455)
            made.append(cap.GetPrim())
        box = UsdGeom.Cube.Define(self.stage, f"{self.root}/panda_hand/xpbd_palm")
        box.CreateSizeAttr(1.0)
        xf = UsdGeom.Xformable(box)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.03))
        xf.AddScaleOp().Set(Gf.Vec3f(0.06, 0.2, 0.06))
        made.append(box.GetPrim())
        for p in made:
            p.CreateAttribute("xpbd:collider", Sdf.ValueTypeNames.Bool).Set(True)
            UsdGeom.Imageable(p).CreateVisibilityAttr("inherited" if visible else "invisible")
            UsdGeom.Gprim(p).CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.55, 0.1)])
        return [str(p.GetPath()) for p in made]


def load_franka(stage, base_pos, yaw_deg=0.0, path="/World/RobotBase"):
    """Franka 를 참조로 넣는다 (자산 루트에 이미 xformOp 가 있어 자리는 부모 Xform 으로). 이미 있으면 자리만 바꾼다."""
    from pxr import Gf, UsdGeom

    base = UsdGeom.Xform.Define(stage, path)
    xf = UsdGeom.Xformable(base)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in base_pos]))
    xf.AddRotateZOp().Set(float(yaw_deg))
    robot = stage.GetPrimAtPath(f"{path}/Franka")
    if not robot or not robot.GetChildren():
        robot = UsdGeom.Xform.Define(stage, f"{path}/Franka").GetPrim()
        robot.GetReferences().AddReference(URL)
    # 자산 기본값(5e-5)이면 팔이 흔들리다 속도 0 인 순간 잠들어 목표에서 떨어진 채 멈춘다 (목표가 그대로면 안 깬다)
    from pxr import Sdf
    robot.CreateAttribute("physxArticulation:sleepThreshold", Sdf.ValueTypeNames.Float).Set(0.0)
    return f"{path}/Franka"
