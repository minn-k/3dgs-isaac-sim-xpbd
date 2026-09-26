"""Public two-arm 3D Gaussian Splatting interaction runtime for Isaac Sim.

Open examples/wolf/wolf_scene.usda, execute run_duo.py from the Script Editor,
and press Play. The runtime creates two Franka arms, attaches selected
Gaussian groups, pulls them outward, twists around the grip axis, and releases.

The loop runs the CUDA solver after each PhysX step, then writes the updated
Gaussian attributes to Fabric once per rendered frame.
"""
import builtins
import json
import os
import sys
import time
from pathlib import Path

import carb.eventdispatcher
import carb.settings
import numpy as np
import omni.kit.app
import omni.physx
import omni.timeline
import omni.usd
from omni.physx.scripts import physicsUtils
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdLux, UsdPhysics
from usdrt import Usd as RtUsd
from usdrt import Vt as RtVt

_SCRIPT_PATH = globals().get("__file__")
if not _SCRIPT_PATH:
    raise RuntimeError("Use src/run_duo.py so the runtime can locate the repository.")
REPO_ROOT = Path(_SCRIPT_PATH).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / "src"
DEMO_DIR = str(REPO_ROOT / "examples")
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))
DATASET_NAME = os.environ.get("APG_3DGS_DATASET", "wolf").strip().lower()
MODEL_SCALE = 1.0
FLOOR_CLEARANCE_M = 0.5
GRAVITY = 5.81
GROUND_SIZE_M = 20.0

# CUDA XPBD solver settings. The graph and Gaussian state are initialized once;
# collision proxies and gripper attachments are refreshed for every PhysX step.
OBJECT_SHAPE = 0.9
FRICTION = 0.5
RESTITUTION = 0.9
DAMPING = 0.01
ITERS = 16
USE_DISTANCE = True
USE_VOLUME = True
USE_SHAPE = False
SHAPE_ROBUST = False
VOL_RING_K = 3
VOL_LEADER_HOP = 2
MODEL_MASS_KG = 1.0
HOLD_WEIGHT = 1.0

# Renderer / timing settings.
REQUIRE_BOUNDED_CACHE = os.environ.get("XPBD_ALLOW_DISK_CACHE") != "1"
CACHE_BUDGET_MAX_MB = 8192
SHAPES = True
DEFORM_EPS = 1e-2
RENDER_MOVE_EPS = 2e-4
SUBSTEPS = 4
SYNC_TIME_CODES = True
KEEP_GRID = True
COLLIDER_EXCLUDE = ()

# Ellipsoid contacts use each Gaussian's current orientation and scale.
CONTACT_SHAPE = 1
CONTACT_TAU = 0.2
CONTACT_CAP_M = 0.02
PAD_MARGIN_M = 0.008
PAD_MARGIN_SHAPE_M = 0.0
SHOW_PADS = False

# Two-arm interaction settings. Arm A selects the top region; Arm B selects a
# lower, outward region. Both arms pull before counter-rotating about the grip axis.
DUO_TASK = "both"                  # "stretch" | "twist" | "both"
DUO_BASE_OFFSETS = ((-0.55, 0.0), (0.60, 0.0))
DUO_GRASP_DEPTH_M = 0.03
DUO_PULL_M = 0.15
DUO_PULL_UP_M = 0.045
DUO_TWIST_DEG = 120.0
DUO_MOVE_S = 4.0
DUO_START_S = 2.5
DUO_OBJECT_SHAPE = 0.05
DUO_LOOP = True
GRASP_SQUEEZE_M = 0.008

_PAD_MARGIN = PAD_MARGIN_SHAPE_M if CONTACT_SHAPE else PAD_MARGIN_M
_KEY = "_gs_duo_runtime"
_GIZMO_KEY = "/exts/omni.kit.hydra_texture/gizmos/enabled"
_MINFR_KEY = "/persistent/simulation/minFrameRate"
def _end_demo(state, tag):
    """Stop callbacks, restore Fabric and USD transforms, then release the CUDA solver."""
    state["sub"] = None
    state["step_sub"] = None
    window = state.pop("ui_window", None)
    state["ui_status"] = None
    if window is not None:
        try:
            window.visible = False
            window.destroy()
        except Exception:  # noqa: BLE001
            pass
    try:
        if "rt_pos" in state and "P0" in state:
            state["rt_pos"].Set(RtVt.Vec3fArray(state["P0"]))
        if "rt_rot" in state and "Q0f" in state:
            state["rt_rot"].Set(RtVt.QuatfArray(state["Q0f"]))
        if "rt_sc" in state and "S0" in state:
            state["rt_sc"].Set(RtVt.Vec3fArray(state["S0"]))
        stage = omni.usd.get_context().get_stage()
        for path in state.get("extra_paths", []):
            if stage.GetPrimAtPath(path):
                stage.RemovePrim(path)
        transform = state.get("model_xf0")
        if transform is not None and stage.GetPrimAtPath(transform[0]):
            prim = stage.GetPrimAtPath(transform[0])
            for name, value in transform[1]:
                prim.GetAttribute(name).Set(value)
        if state.get("min_fr0") is not None:
            carb.settings.get_settings().set(_MINFR_KEY, state["min_fr0"])
        if "sim" in state:
            state["sim"].clear_attached()
            state["sim"].destroy()
    except Exception as exc:  # noqa: BLE001
        print(f"[duo] cleanup {tag}: {exc}")
_prev = getattr(builtins, _KEY, None)
_settings = carb.settings.get_settings()

if _prev is not None:
    # 두 번째 실행: 끝내고 원래대로
    _end_demo(_prev, "duo")
    delattr(builtins, _KEY)
    print("[duo] runtime stopped - original model state restored")
elif REQUIRE_BOUNDED_CACHE and _settings.get("/UJITSO/datastore/allowLocalDataStore") is False:
    print("[duo] NOT STARTED: the renderer cache is switched off (allowLocalDataStore=false) - Gaussians then render as "
          "gray points. Close Isaac Sim and start it with "
          r"scripts\launch_isaac_safe.bat")
elif REQUIRE_BOUNDED_CACHE and (_settings.get("/UJITSO/datastore/localDataStore/largeChunkDiskBudgetMB")
                                or 102400) > CACHE_BUDGET_MAX_MB:
    print("[duo] NOT STARTED: this Isaac Sim keeps up to 100 GB of renderer cache on disk and the deforming Gaussians "
          "fill it (it filled C: before). Close Isaac Sim and start it with "
          r"scripts\launch_isaac_safe.bat")
else:

    import importlib  # noqa: E402

    import splat_io as _ps  # noqa: E402
    import xpbd_bridge as _xm  # noqa: E402
    importlib.reload(_ps)
    importlib.reload(_xm)
    import isaac_colliders as _ic  # noqa: E402
    importlib.reload(_ic)
    from isaac_colliders import ColliderScene  # noqa: E402
    from xpbd_bridge import XPBD, load_inputs  # noqa: E402

    _ctx = omni.usd.get_context()
    _stage = _ctx.get_stage()
    _name = DATASET_NAME
    _Name = _name[:1].upper() + _name[1:]
    _splat_path = f"/World/{_Name}/Splat"
    _prim = _stage.GetPrimAtPath(_splat_path)
    if not _prim or _prim.GetTypeName() != "ParticleField3DGaussianSplat":
        print(f"[duo] {_splat_path} not found - open examples/wolf/wolf_scene.usda first")
    else:
        _d = os.path.join(DEMO_DIR, _name)
        _inp = load_inputs(_d, _name)
        _P0 = _inp["pos"]
        _P_usd = np.array(_prim.GetAttribute("positions").Get(), dtype=np.float32)
        if _P_usd.shape != _P0.shape or not np.array_equal(_P_usd, _P0):
            print("[duo] USD splat positions differ from the physics inputs (order or data) - rebuild the USD")
        else:
            _t0 = time.perf_counter()
            # 원본 좌표 → Isaac 월드: world = s·R·p + t  (R 의 셋째 행 = up_source)
            _xf = json.load(open(os.path.join(_d, f"{_name}_transform.json"), encoding="utf-8"))
            _s = float(_xf["scale"])
            _Rm = np.array(_xf["rotation_matrix"], dtype=np.float64)
            _tw = np.array(_xf["translate"], dtype=np.float64)
            _up = np.array(_xf["up_source"], dtype=np.float64)
            _h = _P0.astype(np.float64) @ _up
            _H = float(np.percentile(_h, 99.8) - np.percentile(_h, 0.2))   # 물체 높이 (원본 단위)
            # Apply the source-to-USD transform, then place the model just above the Isaac ground.
            _c_src = _P0.astype(np.float64).mean(0)
            _cw0 = _s * (_Rm @ _c_src) + _tw
            _pl0 = ColliderScene(_stage, np.eye(3), np.zeros(3), 1.0, exclude=[f"/World/{_Name}"]).discover().planes
            _floor_z = float(np.array(UsdGeom.Xformable(_pl0[0]).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()))[3, 2]) if _pl0 else 0.0
            _model_prim = _stage.GetPrimAtPath(f"/World/{_Name}")
            _a_tr = _model_prim.GetAttribute("xformOp:translate")
            _a_sc = _model_prim.GetAttribute("xformOp:scale")
            if not (_a_tr and _a_sc):
                raise RuntimeError(f"/World/{_Name} has no xformOp:translate / xformOp:scale; rebuild the scene")
            _model_xf0 = (str(_model_prim.GetPath()), [
                ("xformOp:translate", Gf.Vec3d(*[float(v) for v in _tw])),
                ("xformOp:scale", Gf.Vec3d(_s, _s, _s)),
            ])
            _s = _s * MODEL_SCALE
            _rc = _s * (_Rm @ _c_src)
            _tw = np.array([_cw0[0] - _rc[0], _cw0[1] - _rc[1], _floor_z + FLOOR_CLEARANCE_M - _s * _h.min()])
            _a_tr.Set(Gf.Vec3d(*[float(v) for v in _tw]))
            _a_sc.Set(Gf.Vec3d(_s, _s, _s))
            _cw = _s * (_Rm @ _c_src) + _tw
            _extra = []

            # ── Isaac 물리 장면: 바닥·중력·충돌 물체는 전부 Isaac 쪽에서 읽는다 (가우시안만 우리 XPBD) ──
            if not any(_p.IsA(UsdPhysics.Scene) for _p in _stage.Traverse()):
                _pscene = UsdPhysics.Scene.Define(_stage, "/World/PhysicsScene")
                _pscene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
                _pscene.CreateGravityMagnitudeAttr(float(GRAVITY))
                _extra.append("/World/PhysicsScene")
            # 시간 맞추기: Isaac GUI 는 한 프레임에 1/timeCodesPerSecond 초를 재생한다 (고정 간격). 이 값이 없는 USD 는 24.
            # PhysX 는 1/60 s 스텝을 그만큼 밟고, XPBD 는 PhysX 스텝마다 같이 간다 → 60/SUBSTEPS 로 두면 프레임당 SUBSTEPS 스텝.
            _tcps_want = 60.0 / max(1, int(SUBSTEPS))
            _tcps = _stage.GetTimeCodesPerSecond()
            if SYNC_TIME_CODES and abs(_tcps - _tcps_want) > 1e-6:
                _stage.SetTimeCodesPerSecond(_tcps_want)
                print(f"[duo] stage timeCodesPerSecond {_tcps:g} -> {_tcps_want:g} ({SUBSTEPS} physics steps per frame)")
            if SYNC_TIME_CODES:
                omni.timeline.get_timeline_interface().set_time_codes_per_second(_tcps_want)
            # PhysX 는 한 프레임에 스텝을 60/minFrameRate 번까지만 밟는다 — 모자라면 잠시 낮춘다 (끝낼 때 되돌린다)
            _min_fr0 = _settings.get(_MINFR_KEY)
            if _min_fr0 is not None and float(_min_fr0) > _tcps_want + 1e-6:
                _settings.set(_MINFR_KEY, type(_min_fr0)(_tcps_want))
                print(f"[duo] PhysX minFrameRate {_min_fr0} -> {_settings.get(_MINFR_KEY)} (up to {SUBSTEPS} steps per frame)")
            else:
                _min_fr0 = None
            for _p in _stage.Traverse():
                if _p.IsA(UsdPhysics.Scene):
                    PhysxSchema.PhysxSceneAPI.Apply(_p).CreateTimeStepsPerSecondAttr(60)
            _scene = ColliderScene(_stage, _Rm, _tw, _s, exclude=[f"/World/{_Name}"] + list(COLLIDER_EXCLUDE)).discover()
            # Create an Isaac ground plane only when the opened scene has no collision plane.
            if not _scene.planes:
                _gp = physicsUtils.add_ground_plane(
                    _stage, "/World/GroundPlane", "Z", float(0.5 * GROUND_SIZE_M),
                    Gf.Vec3f(float(_cw[0]), float(_cw[1]), float(_floor_z)), Gf.Vec3f(0.55, 0.55, 0.58))
                _extra.append(str(_gp))
                _scene.discover()
            _z_floor = float(np.array(UsdGeom.Xformable(_scene.planes[0]).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()))[3, 2])                               # 바닥 월드 높이 (큐브 동작·로그용)

            _sim = XPBD()
            _sim.create(_inp)
            _sim.set_solver(iters=ITERS, dt=1 / 60, under_relax=0.6, vel_damping=DAMPING)
            _sim.set_constraints(distance=USE_DISTANCE, shape=USE_SHAPE, angle=False, volume=USE_VOLUME)
            _sim.set_volume(compliance=1e-6, ring_k=VOL_RING_K, max_members=2048, leader_min_hop=VOL_LEADER_HOP)
            _sim.set_shape_robust(SHAPE_ROBUST)
            _sim.set_object_shape(OBJECT_SHAPE)
            _sim.set_object_shape_gpu(True)
            _contact_on = bool(CONTACT_SHAPE) and _sim.set_contact_shape(int(CONTACT_SHAPE), CONTACT_TAU, CONTACT_CAP_M / _s)
            if CONTACT_SHAPE and not _contact_on:
                print("[duo] this xpbd DLL has no ellipsoid contact - using center points (rebuild xpbd_dll)")
                _PAD_MARGIN = PAD_MARGIN_M
            _sim.set_ground(False, _up, 0.0, gravity=0.0)
            _sim.step()    # 그래프·부피 클러스터를 GPU 에 올린다 (중력 없이 — 움직이지 않는다)
            _sim.reset()
            _tilt = (0.0, 0.0, 0.0)

            def _scene_ground(scene, _ref=_P0.astype(np.float64).mean(0), _Hs=_H):
                """Isaac 바닥·중력 → DLL set_ground 인자 (새 튜플 = 바뀜). 바닥이 없으면 중력만 (바닥 끔)."""
                gdir, gmag = scene.gravity(GRAVITY)
                g = scene.ground(_ref)
                if g is None:
                    return (False, tuple(-gdir), 0.0, gmag)
                up, h = g
                if float(up @ -gdir) < 0.996:
                    print("[duo] WARNING: gravity is not perpendicular to the ground plane - the Gaussians fall along "
                          "the plane normal")
                return (True, tuple(up), h, gmag)

            # Two Franka arms and their XPBD-only fingertip / palm collision proxies.
            import robot_arm as _ra  # noqa: E402
            import robot_duo as _rd  # noqa: E402
            from robot_common import FINGER_OPEN, grasp_box, grip_opening, rigid_fit  # noqa: E402
            importlib.reload(_ra)
            importlib.reload(_rd)
            _duo_arms, _duo = [], None
            for _k, (_ox, _oy) in enumerate(DUO_BASE_OFFSETS):
                _base = "/World/RobotBase" if _k == 0 else f"/World/RobotBase{_k + 1}"
                _robot = _ra.load_franka(
                    _stage,
                    (_cw[0] + _ox, _cw[1] + _oy, _z_floor),
                    yaw_deg=float(np.degrees(np.arctan2(-_oy, -_ox))),
                    path=_base,
                )
                _extra.append(_base)
                if not _stage.GetPrimAtPath(f"{_robot}/panda_hand"):
                    raise RuntimeError("The Franka asset did not load. Check Isaac Sim asset access and run the script again.")
                _arm = _ra.FrankaArm(_stage, _robot)
                _arm.add_xpbd_pads(visible=SHOW_PADS, pad_radius=0.0095 + _PAD_MARGIN)
                _arm.set_state(_arm.home, finger=0.0)
                _duo_arms.append(_arm)
            _duo = _rd.DuoDemo(
                _duo_arms,
                task=DUO_TASK,
                grasp_depth=DUO_GRASP_DEPTH_M,
                squeeze=GRASP_SQUEEZE_M,
                pad_r=0.0095 + _PAD_MARGIN,
                pull=DUO_PULL_M,
                pull_up=DUO_PULL_UP_M,
                twist_deg=DUO_TWIST_DEG,
                speed_s=DUO_MOVE_S,
                floor_z=_z_floor,
            )
            _scene.discover()
            print(
                f"[duo] Franka A (top) at {np.round(_duo_arms[0].base[:3, 3], 3).tolist()}, "
                f"Franka B (lower end) at {np.round(_duo_arms[1].base[:3, 3], 3).tolist()}, "
                f"{_name}; object shape while both hands hold: {DUO_OBJECT_SHAPE} (rest: {OBJECT_SHAPE})"
            )
            # 바닥 판은 빛이 있어야 보인다 (가우시안은 조명과 무관). 여기서 만든 빛만 끝낼 때 지운다.
            if not any(p.IsA(UsdLux.BoundableLightBase) or p.IsA(UsdLux.NonboundableLightBase) for p in _stage.Traverse()):
                UsdLux.DomeLight.Define(_stage, "/World/DemoDomeLight").CreateIntensityAttr(1000.0)
                _extra.append("/World/DemoDomeLight")

            _rt_stage = RtUsd.Stage.Attach(_ctx.get_stage_id())
            _rt_prim = _rt_stage.GetPrimAtPath(_splat_path)
            _rt_pos = _rt_prim.GetAttribute("positions")
            _rt_rot = _rt_prim.GetAttribute("orientations")
            _rt_sc = _rt_prim.GetAttribute("scales")
            # Fabric 쿼터니언 메모리 순서: 장면 값과 PLY rot (w,x,y,z) 를 비교해 정한다 (Isaac 6.1 실측: x,y,z,w)
            _Q0f = np.ascontiguousarray(np.array(_rt_rot.Get(), dtype=np.float32))
            _S0 = np.ascontiguousarray(np.array(_rt_sc.Get(), dtype=np.float32))
            _qw = _inp["rots"]
            _err_wxyz = float(np.abs(np.abs((_Q0f * _qw).sum(1)) - 1).max())
            _err_xyzw = float(np.abs(np.abs((_Q0f * np.concatenate([_qw[:, 1:], _qw[:, :1]], axis=1)).sum(1)) - 1).max())
            _quat_xyzw = _err_xyzw < _err_wxyz
            if min(_err_wxyz, _err_xyzw) > 1e-3:
                print(f"[duo] WARNING: cannot match Fabric orientation layout (errors {_err_wxyz:.2e}, {_err_xyzw:.2e}) - "
                      f"shapes disabled")
                SHAPES = False

            _st = {
                "sim": _sim,
                "P0": _P0,
                "rt_pos": _rt_pos,
                "rt_rot": _rt_rot,
                "rt_sc": _rt_sc,
                "Q0f": _Q0f,
                "S0": _S0,
                "sub": None,
                "step_sub": None,
                "extra_paths": _extra,
                "model_xf0": _model_xf0,
                "min_fr0": _min_fr0,
                "phase": "idle",
                "stop_wait": 0,
                "steps": 0,
                "new_steps": 0,
                "dead": False,
                "frames": 0,
                "t_step": 0.0,
                "t_io": 0.0,
                "wall0": time.perf_counter(),
                "steps0": 0,
                "next_log": 60,
                "P_last": _P0,
                "writes": 0,
                "fab": None,
                "rewrite_in": [],
                "idle_frames": 0,
                "scene": _scene,
                "ground": _scene_ground(_scene),
                "ground_applied": None,
                "col_hits": None,
                "model_velocity": np.zeros(3),
                "com_prev": None,
                "duo": {
                    "planned": False,
                    "mode": "auto",
                    "ui_request": None,
                    "ui_cfg": {
                        "pull": DUO_PULL_M,
                        "lift": DUO_PULL_UP_M,
                        "twist": DUO_TWIST_DEG,
                        "speed": DUO_MOVE_S,
                        "offsets": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    },
                },
                "grasps": [None, None],
            }

            # 콜백이 쓰는 값은 정의 시점에 묶어 둔다 — Script Editor 의 전역 이름은 다른 스크립트가 덮어쓸 수 있다.
            def _write_fabric(st, P, sc, q):
                st["rt_pos"].Set(RtVt.Vec3fArray(P))
                if sc is not None:
                    st["rt_rot"].Set(RtVt.QuatfArray(q))
                    st["rt_sc"].Set(RtVt.Vec3fArray(sc))
                st["fab"] = (P, sc, q)

            def _read_shape(sim, _shapes=SHAPES, _eps=DEFORM_EPS, _xyzw=_quat_xyzw):
                """GPU 결과 → CPU: 위치와 (SHAPES 면) 변형에 맞춘 크기·방향 (Fabric 쿼터니언 순서로)."""
                P = sim.positions()
                sc = q = None
                if _shapes:
                    sc, q, _k = sim.shapes(_eps)          # q = (w, x, y, z)
                    if _xyzw:
                        q = np.ascontiguousarray(np.concatenate([q[:, 1:], q[:, :1]], axis=1))
                return P, sc, q

            def _reset(state, _read=_read_shape):
                """Restore the input 3DGS state and park both arms at their home pose."""
                sim = state["sim"]
                sim.reset()
                sim.launch((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
                sim.clear_attached()
                for arm in _duo_arms:
                    arm.set_state(arm.home, finger=0.0)
                state["steps"] = 0
                state["new_steps"] = 0
                state["dead"] = False
                state["model_velocity"] = np.zeros(3)
                state["com_prev"] = None
                state["ground_applied"] = None
                state["col_hits"] = None
                state["duo"] = {
                    "planned": False,
                    "mode": "auto",
                    "ui_request": None,
                    "ui_cfg": {
                        "pull": DUO_PULL_M,
                        "lift": DUO_PULL_UP_M,
                        "twist": DUO_TWIST_DEG,
                        "speed": DUO_MOVE_S,
                        "offsets": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    },
                }
                state["grasps"] = [None, None]
                sim.set_object_shape(OBJECT_SHAPE)
                positions, scales, rotations = _read(sim)
                _write_fabric(state, positions, scales, rotations)
                state["phase"] = "idle"
                state["P_last"] = positions
                state["rewrite_in"] = [5, 20]
                state["idle_frames"] = 0

            def _start(state, _scene_ground=_scene_ground):
                """Refresh PhysX collision proxies and synchronize the XPBD ground state."""
                state["scene"].discover()
                state["ground"] = _scene_ground(state["scene"])
                state["phase"] = "running"
                state["rewrite_in"] = []
                state["frames"] = 0
                state["t_step"] = 0.0
                state["t_io"] = 0.0
                state["writes"] = 0
                state["wall0"] = time.perf_counter()
                state["steps0"] = state["steps"]
                state["next_log"] = state["steps"] + 60
                print(f"[duo] playing: {SUBSTEPS} PhysX / XPBD steps per rendered frame; {state['scene'].summary()}")
            def _hand_world(hand_path):
                """Read the current PhysX hand pose as a column-vector 4 x 4 transform."""
                result = omni.physx.get_physx_interface().get_rigidbody_transformation(hand_path)
                if not result.get("ret_val"):
                    return None
                x, y, z, w = result["rotation"]
                transform = np.eye(4)
                transform[:3, :3] = _ra.quat_wxyz_to_mat((w, x, y, z))
                transform[:3, 3] = result["position"]
                return transform
            _duo_hands = [f"{_a.root}/panda_hand" for _a in _duo_arms]

            def _duo_grab(st, P, k, _s=_s, _Rm=_Rm, _tw=_tw):
                """duo: 손 k 의 두 손가락 사이 가우시안을 그 손에 붙인다 (다른 손이 이미 잡은 것은 뺀다)."""
                Hw = _hand_world(_duo_hands[k])
                if Hw is None:
                    return 0
                W = _s * (P.astype(np.float64) @ _Rm.T) + _tw
                idx, local = grasp_box(W, Hw, _duo_arms[k].finger_state())
                other = st["grasps"][1 - k]
                if other is not None and len(idx):
                    keep = ~np.isin(idx, other["idx"])
                    idx, local = idx[keep], local[keep]
                if len(idx) < 15:
                    print(f"[duo] hand {'AB'[k]}: only {len(idx)} Gaussians between the fingers - nothing to hold")
                    return 0
                st["grasps"][k] = {"idx": idx.astype(np.int32), "local": local}
                if all(g is not None for g in st["grasps"]):
                    st["sim"].set_object_shape(DUO_OBJECT_SHAPE)
                return len(idx)

            def _duo_release(st):
                if any(g is not None for g in st["grasps"]):
                    st["sim"].clear_attached()
                    st["grasps"] = [None, None]
                    st["sim"].set_object_shape(OBJECT_SHAPE)
                    print("[duo] released (both hands)")

            def _duo_hold_step(st, _s=_s, _Rm=_Rm, _tw=_tw, _n=len(_P0)):
                """물리 스텝마다: 두 손이 잡은 가우시안을 각 손의 PhysX 자세로 고정 (한 번에 넘긴다), 무게는 두 손에 나눠 건다."""
                gs = [(k, g) for k, g in enumerate(st["grasps"]) if g is not None]
                if not gs:
                    return
                import carb
                from pxr import PhysicsSchemaTools

                idx, pos, hands = [], [], []
                for k, g in gs:
                    Hw = _hand_world(_duo_hands[k])
                    if Hw is None:
                        return
                    idx.append(g["idx"])
                    pos.append(g["local"] @ Hw[:3, :3].T + Hw[:3, 3])
                    hands.append((k, Hw))
                idx = np.concatenate(idx)
                w = max(1.0, HOLD_WEIGHT * (_n - len(idx)) / len(idx))
                st["sim"].set_attached(idx, (np.concatenate(pos) - _tw) @ _Rm / _s, w)
                gmag = st["ground"][3] * _s
                sim_if = omni.physx.get_physx_simulation_interface()
                for k, Hw in hands:
                    sim_if.apply_force_at_pos(omni.usd.get_context().get_stage_id(),
                                              PhysicsSchemaTools.sdfPathToInt(_duo_hands[k]),
                                              carb.Float3(0.0, 0.0, float(-MODEL_MASS_KG * gmag / len(hands))),
                                              carb.Float3(*[float(v) for v in Hw[:3, 3]]), "Force")

            def _step(state, _height=_H, _col_margin=0.003 * _H):
                """Advance the CUDA solver once after each 60 Hz PhysX step."""
                t0 = time.perf_counter()
                sim = state["sim"]
                ground = state.get("ground")
                if ground is not None and ground != state.get("ground_applied"):
                    enabled, up, height, gravity = ground
                    sim.set_ground(
                        enabled,
                        up,
                        height,
                        friction=FRICTION,
                        restitution=RESTITUTION,
                        contact_radius=0.003 * _height,
                        contact_slop=0.002 * _height,
                        gravity=gravity,
                    )
                    state["ground_applied"] = ground
                if state["steps"] % 60 == 59:
                    state["scene"].discover()
                colliders = state["scene"].colliders()
                sim.set_colliders(*colliders, margin=_col_margin, friction=FRICTION)
                _duo_hold_step(state)
                sim.step()
                state["col_hits"] = None
                if len(colliders[0]):
                    state["col_hits"] = sim.collider_stats(len(colliders[0]))[0]
                state["steps"] += 1
                state["new_steps"] += 1
                state["t_step"] += time.perf_counter() - t0
            _reset(_st)

            def _rot_about_axis_deg(A, B, axis):
                """점 묶음 A → B 강체 맞춤 회전 중 기준축 성분 [deg]."""
                R, _t = rigid_fit(A, B)
                q = np.asarray(_ra.mat_to_quat_wxyz(R), float)
                if q[0] < 0.0:
                    q = -q
                return float(np.degrees(2.0 * np.arctan2(np.dot(q[1:], axis / np.linalg.norm(axis)), q[0])))

            def _duo_metrics(st, W, _W0=_Pw0):
                """duo: 잡은 수, 두 잡은 곳 사이 거리 변화, 머리 쪽·가운데·발 쪽의 그립축 회전(잡은 순간 대비),
                rest 모양 대비 변형 p95 (강체 맞춤 뒤), 손끝 오차."""
                d = st["duo"]
                out = {"held": [0 if g is None else len(g["idx"]) for g in st["grasps"]]}
                R, t = rigid_fit(_W0, W)
                out["deform"] = float(np.percentile(np.linalg.norm(W - (_W0 @ R.T + t), axis=1), 95) * 1000.0)
                ref = d.get("W_grab")
                if ref is not None:
                    ca, cb = W[d["reg_a"]].mean(0), W[d["reg_b"]].mean(0)
                    out["span"] = float(np.linalg.norm(ca - cb) - d["span0"]) * 1000.0
                    axis = np.asarray(d.get("grip_axis", [0.0, 0.0, 1.0]), float)
                    out["rot"] = [(_rot_about_axis_deg(ref[d[r]], W[d[r]], axis) if len(d[r]) >= 3 else float("nan"))
                                  for r in ("reg_a", "reg_mid", "reg_b")]
                else:
                    out["span"], out["rot"] = float("nan"), [float("nan")] * 3
                errs = []
                for k, a in enumerate(_duo_arms):
                    Hw, pc = _hand_world(_duo_hands[k]), d.get("p_cmd", [None, None])[k]
                    errs.append(float("nan") if Hw is None or pc is None else
                                float(np.linalg.norm(Hw[:3, 3] + _ra.TCP_OFFSET * Hw[:3, 2] - pc) * 1000.0))
                out["tip_err"] = errs
                return out

            def _duo_enter_manual(st):
                """자동 경로를 중단하고 두 TCP 의 현재 자세를 기준으로 수동 조종을 시작한다."""
                d = st["duo"]
                if d.get("mode") == "manual" and d.get("servos"):
                    return
                qs = [a.joint_state() for a in _duo_arms]
                servos = [_rd.HandServo(a, q, v_max=0.12, w_max_deg=60.0) for a, q in zip(_duo_arms, qs)]
                d["servos"] = servos
                d["manual_origin"] = [s.p.copy() for s in servos]
                d["manual_R"] = [s.R.copy() for s in servos]
                d["manual_hold"] = [({"p": s.p.copy(), "R": s.R.copy()} if st["grasps"][k] is not None else None)
                                     for k, s in enumerate(servos)]
                d["manual_grab_wait"] = [False, False]
                d["manual_reanchor_pending"] = [False, False]
                d["last_manual_t"] = None
                d["q_end"] = [s.q.copy() for s in servos]
                d["planned"], d["single_done"], d["mode"] = False, False, "manual"
                print("[duo] manual control: use the Two-arm control window")

            def _duo_manual_action(st, req, P, t_sim, cfg, _s=_s, _Rm=_Rm, _tw=_tw):
                """버튼 입력을 물리 스텝에서 처리해 TCP 목표나 파지 상태를 갱신한다."""
                d = st["duo"]
                _duo_enter_manual(st)
                servos = d["servos"]
                manual_span = max(cfg["pull"], cfg["lift"],
                                  max((abs(v) for row in cfg["offsets"] for v in row), default=0.0), 0.04)
                for servo in servos:
                    servo.v_max = max(0.01, min(0.20, manual_span / max(cfg["speed"], 0.5)))
                    servo.w_max = max(np.radians(5.0), np.radians(min(abs(cfg["twist"]), 180.0)) /
                                      max(cfg["speed"], 0.5))
                if req == "approach_grab":
                    W = _s * (P.astype(np.float64) @ _Rm.T) + _tw
                    _duo_release(st)
                    d["manual_grab_wait"] = [False, False]
                    for arm, servo in zip(_duo_arms, servos):
                        servo.finger = FINGER_OPEN
                        arm.set_targets(servo.q, FINGER_OPEN)
                    _duo.plan(W, [servo.q for servo in servos])
                    for k, (servo, path) in enumerate(zip(servos, _duo.paths)):
                        steps = path.segments[:3]  # move over -> descend -> close
                        targets = [(seg[2], seg[3], seg[4]) for seg in steps]
                        servo.go(targets[-1][0], targets[-1][1], targets[-1][2], via=targets[:-1])
                        d["manual_hold"][k] = None
                        d["manual_grab_wait"][k] = True
                    print("[duo] manual approach and grasp started: A=top, B=lower end; will stop after both close")
                    return
                if req == "grab_both":
                    hands = (0, 1)
                elif req == "grab_a":
                    hands = (0,)
                elif req == "grab_b":
                    hands = (1,)
                else:
                    hands = ()
                if hands:
                    d["manual_grab_wait"] = [False, False]
                    W = _s * (P.astype(np.float64) @ _Rm.T) + _tw
                    for k in hands:
                        Hw = _hand_world(_duo_hands[k])
                        if Hw is None:
                            continue
                        q_grip, n = grip_opening(W, Hw, _duo_arms[k].finger_state(),
                                                      0.0095 + _PAD_MARGIN, GRASP_SQUEEZE_M)
                        servos[k].go(servos[k].p, servos[k].R, q_grip)
                        d["manual_grab_wait"][k] = True
                        print(f"[duo] closing hand {'AB'[k]} to {q_grip * 1000:.1f} mm; attach when closed "
                              f"({n} candidate Gaussians)")
                    return
                if req == "release":
                    _duo_release(st)
                    d["manual_hold"] = [None, None]
                    d["manual_grab_wait"] = [False, False]
                    d["manual_reanchor_pending"] = [False, False]
                    for servo in servos:
                        servo.go(servo.p, servo.R, FINGER_OPEN)
                    return
                if req in ("move_a", "move_b", "move_both"):
                    hands = (0,) if req == "move_a" else (1,) if req == "move_b" else (0, 1)
                    for k in hands:
                        d["manual_grab_wait"][k] = False
                        d["manual_reanchor_pending"][k] = st["grasps"][k] is not None
                        off = np.asarray(cfg["offsets"][k], float)
                        p = d["manual_origin"][k] + off
                        p[2] = max(p[2], _z_floor + 0.012)
                        servos[k].go(p, d["manual_R"][k], servos[k].finger)
                    return
                if req in ("pull", "twist", "pull_twist", "relax", "reset_grip"):
                    holds = d.get("manual_hold", [None, None])
                    if not all(h is not None for h in holds) or not all(g is not None for g in st["grasps"]):
                        print("[duo] manual pull/twist needs both hands gripping; press Grab both first")
                        return
                    if not all(servo.arrived() for servo in servos):
                        print("[duo] wait for both hands to finish their current move before starting another pull/twist action")
                        return
                    gp = {"grasp": [h["p"] for h in holds], "R0": [h["R"] for h in holds],
                          "pull_dir": [None, None]}
                    delta = gp["grasp"][1] - gp["grasp"][0]
                    delta[2] = 0.0
                    norm = float(np.linalg.norm(delta))
                    if norm < 1e-5:
                        print("[duo] cannot determine the pull axis because the hands are too close")
                        return
                    axis_xy = delta / norm
                    gp["pull_dir"] = [-axis_xy, axis_xy]
                    if req == "relax":
                        pull, lift, twist = 0.0, 0.005, 0.0
                    elif req == "reset_grip":
                        pull, lift, twist = 0.0, 0.0, 0.0
                    else:
                        pull, lift = cfg["pull"], cfg["lift"]
                        twist = cfg["twist"] if req in ("twist", "pull_twist") else 0.0
                    asked = (pull, lift, twist)
                    pull, lift, twist = _duo.reach_limits(gp, [servo.q for servo in servos], pull, lift, twist)
                    if (abs(pull - asked[0]) > 1e-4 or abs(twist - asked[2]) > 1e-3):
                        print(f"[duo] out of reach: pull {asked[0] * 100:.0f} -> {pull * 100:.0f} cm per hand, twist "
                              f"{asked[2]:.0f} -> {twist:.0f} deg (the arms cannot reach further from here)")
                    d["manual_used"] = (pull, lift, twist)
                    ps, Rs = _duo.pose_at(gp, pull, lift, twist)
                    if req == "pull_twist" and abs(twist) > 1e-4:
                        p_pull, R_pull = _duo.pose_at(gp, pull, lift, 0.0)
                        for k, servo in enumerate(servos):
                            servo.go(ps[k], Rs[k], servo.finger,
                                     via=[(p_pull[k], R_pull[k], servo.finger)])
                    else:
                        for k, servo in enumerate(servos):
                            servo.go(ps[k], Rs[k], servo.finger)
                    return

            def _duo_manual_step(st, P, t_sim, _s=_s, _Rm=_Rm, _tw=_tw):
                d = st["duo"]
                servos = d.get("servos") or []
                if not servos:
                    _duo_enter_manual(st)
                    servos = d["servos"]
                dt = 1.0 / 60.0 if d.get("last_manual_t") is None else max(1.0 / 240.0, t_sim - d["last_manual_t"])
                d["last_manual_t"] = t_sim
                for k, (arm, servo) in enumerate(zip(_duo_arms, servos)):
                    q, finger, p_cmd, _ep = servo.step(dt)
                    arm.set_targets(q, finger)
                    d["q_end"] = [s.q.copy() for s in servos]
                    if d.get("p_cmd") is None:
                        d["p_cmd"] = [None, None]
                    d["p_cmd"][k] = p_cmd
                    if d.get("manual_err") is None:
                        d["manual_err"] = [0.0, 0.0]
                    d["manual_err"][k] = float(_ep)
                    if d.get("manual_grab_wait", [False, False])[k] and servo.arrived():
                        d["manual_grab_wait"][k] = False
                        n = _duo_grab(st, P, k)
                        if n:
                            d["manual_hold"][k] = {"p": servo.p.copy(), "R": servo.R.copy()}
                        else:
                            print(f"[duo] hand {'AB'[k]} did not grip enough Gaussians; reposition and retry")
                    if d.get("manual_reanchor_pending", [False, False])[k] and servo.arrived():
                        d["manual_hold"][k] = {"p": servo.p.copy(), "R": servo.R.copy()}
                        d["manual_reanchor_pending"][k] = False
                lab = st.get("ui_status")
                if lab is not None:
                    held = [0 if g is None else len(g["idx"]) for g in st["grasps"]]
                    err = d.get("manual_err", [0.0, 0.0])
                    lab.text = (f"MANUAL | A {held[0]} / B {held[1]} held | IK {err[0] * 1000:.0f}/{err[1] * 1000:.0f} mm | "
                                f"pull {d['ui_cfg']['pull'] * 1000:.0f} mm/hand | twist {d['ui_cfg']['twist']:.0f} deg")

            def _duo_step(st, P, t_sim, _s=_s, _Rm=_Rm, _tw=_tw):
                """duo 자동/수동: 물체가 자리 잡으면 계획 경로를 수행하거나 창의 버튼 요청을 두 팔에 적용한다."""
                d = st["duo"]
                request, d["ui_request"] = d.get("ui_request"), None
                cfg = d.get("ui_cfg", {"pull": DUO_PULL_M, "lift": DUO_PULL_UP_M, "twist": DUO_TWIST_DEG,
                                        "speed": DUO_MOVE_S, "offsets": [[0.0] * 3, [0.0] * 3]})
                if request is not None:
                    req, cfg = request
                    d["ui_cfg"] = cfg
                    _duo.pull, _duo.pull_up = cfg["pull"], cfg["lift"]
                    _duo.twist_deg, _duo.speed_s = cfg["twist"], cfg["speed"]
                    if req == "manual":
                        _duo_enter_manual(st)
                    elif req in ("auto_loop", "run_once"):
                        q_now = [a.joint_state() for a in _duo_arms]
                        _duo_release(st)
                        for a, q in zip(_duo_arms, q_now):
                            a.set_targets(q, FINGER_OPEN)
                        d.update(mode="auto", servos=None, planned=False, single_done=False, abort_reason=None,
                                 loop=(req == "auto_loop"), q_end=q_now)
                        print(f"[duo] {'automatic loop' if req == 'auto_loop' else 'one automatic cycle'} requested; "
                              f"pull {cfg['pull'] * 100:.0f} cm/hand, lift {cfg['lift'] * 100:.1f} cm, twist {cfg['twist']:.0f} deg")
                    else:
                        _duo_manual_action(st, req, P, t_sim, cfg)
                if d.get("mode") == "manual":
                    _duo_manual_step(st, P, t_sim)
                    return
                if d.get("single_done"):
                    lab = st.get("ui_status")
                    if lab is not None:
                        lab.text = (f"GRASP FAILED: {d['abort_reason']}" if d.get("abort_reason") else
                                    f"AUTO (one cycle complete) | pull {cfg['pull'] * 1000:.0f} mm/hand | twist {cfg['twist']:.0f} deg")
                    return
                if not d["planned"]:
                    if t_sim < DUO_START_S or float(np.linalg.norm(st["model_velocity"])) > 0.05:
                        lab = st.get("ui_status")
                        if lab is not None:
                            lab.text = "AUTO | waiting for the model to settle"
                        return
                    W = _s * (P.astype(np.float64) @ _Rm.T) + _tw
                    q_now = d.get("q_end") or [a.home for a in _duo_arms]
                    info = _duo.plan(W, q_now)
                    d.update(planned=True, t0=t_sim, phase=None, rows=[], W_grab=None)
                    reach = 0.0     # 실행과 같이 앞 구간 자세에서 이어 푼 IK 의 가장 큰 손끝 오차 (기본 자세에서 새로 풀면
                    for a, path in zip(_duo_arms, _duo.paths):        # 다른 해로 가 없는 오차가 나온다)
                        qq = path.q.copy()
                        for (_n, _dd, p, R, _f) in path.segments:
                            qq, _e, _er = a.ik(p[-1] if p.ndim == 2 else p, R, qq, iters=40)
                            reach = max(reach, _e * 1000.0)
                    print(f"[duo] plan at t {t_sim:.2f} s: A grasps the top at {np.round(info['grasp'][0], 3).tolist()} "
                          f"(fingers {info['grip'][0][0] * 1000:.1f} mm), B grasps the front-low part at "
                          f"{np.round(info['grasp'][1], 3).tolist()} (fingers {info['grip'][1][0] * 1000:.1f} mm), "
                          f"{len(_duo.paths[0].segments)} moves over {_duo.duration:.1f} s, worst IK reach error {reach:.1f} mm"
                          + ("  <- out of reach: move DUO_BASE_OFFSETS" if reach > 10.0 else ""))   # reach_limits 허용 8 mm
                    _u = info.get("used")
                    if _u is not None:
                        _a = _u["asked"]
                        print(f"[duo] this cycle: pull {_u['pull'] * 100:.0f} cm per hand, lift {_u['lift'] * 100:.1f} cm, "
                              f"twist {_u['twist']:.0f} deg about the grip axis"
                              + (f"  (asked {_a[0] * 100:.0f} cm / {_a[2]:.0f} deg - reduced to what both arms reach)"
                                 if abs(_u["pull"] - _a[0]) > 1e-4 or abs(_u["twist"] - _a[2]) > 1e-3 else ""))
                tl = t_sim - d["t0"]
                cmds = _duo.command(tl)
                for a, (q, fin, _nm, _p, _e) in zip(_duo_arms, cmds):
                    a.set_targets(q, fin)
                d["p_cmd"] = [c[3] for c in cmds]
                name = cmds[0][2]
                if name != d["phase"]:
                    prev = d["phase"]
                    if prev is not None:
                        W = _s * (P.astype(np.float64) @ _Rm.T) + _tw
                        if prev == "close":
                            n = [_duo_grab(st, P, k) for k in (0, 1)]
                            if not all(n):
                                _duo_release(st)
                                q_stop = [c[0] for c in cmds]
                                for arm, q in zip(_duo_arms, q_stop):
                                    arm.set_targets(q, FINGER_OPEN)
                                d.update(q_end=q_stop, phase="grasp failed", single_done=True,
                                         abort_reason=f"A {n[0]} / B {n[1]} Gaussians")
                                print(f"[duo] STOP: both-end grasp required, but attached {n[0]} / {n[1]} Gaussians. "
                                      "Reposition the robots or adjust DUO_GRASP_DEPTH_M, then retry.")
                                lab = st.get("ui_status")
                                if lab is not None:
                                    lab.text = f"GRASP FAILED | A {n[0]} / B {n[1]}; reposition and retry"
                                return
                            # 수치용 영역: 두 잡은 곳 둘레 4 cm, 그 사이 선분 가운데 40~60% 에서 선 둘레 6 cm (몸통)
                            ga, gb = W[st["grasps"][0]["idx"]].mean(0) if n[0] else None, \
                                W[st["grasps"][1]["idx"]].mean(0) if n[1] else None
                            if n[0] and n[1]:
                                ab = gb - ga
                                u = (W - ga) @ ab / float(ab @ ab)
                                off = np.linalg.norm(W - (ga + u[:, None] * ab), axis=1)
                                d.update(W_grab=W, grip_axis=ab / np.linalg.norm(ab),
                                         reg_a=np.where(np.linalg.norm(W - ga, axis=1) < 0.04)[0],
                                         reg_b=np.where(np.linalg.norm(W - gb, axis=1) < 0.04)[0],
                                         reg_mid=np.where((u > 0.4) & (u < 0.6) & (off < 0.06))[0],
                                         span0=float(np.linalg.norm(ga - gb)))
                            print(f"[duo] grabbed A {n[0]} / B {n[1]} Gaussians"
                                  + (f", {np.linalg.norm(ga - gb) * 100:.1f} cm apart, middle band "
                                     f"{len(d['reg_mid'])} Gaussians" if n[0] and n[1] else " - not both hands: skip"))
                        m = _duo_metrics(st, W)
                        print(f"[duo] {tl:5.2f} s  end of '{prev.strip()}': holding A {m['held'][0]} / B {m['held'][1]} | "
                              f"grip span {m['span']:+6.1f} mm | rotation about grip axis grip A {m['rot'][0]:+6.1f}, middle "
                              f"{m['rot'][1]:+6.1f}, grip B {m['rot'][2]:+6.1f} deg | deformation vs rest p95 {m['deform']:5.1f} mm"
                              f" | tip error A {m['tip_err'][0]:4.1f} B {m['tip_err'][1]:4.1f} mm")
                        d["rows"].append((prev.strip(), tl, m))
                    if name == "open":
                        _duo_release(st)
                    d["phase"] = name
                if name == "done":
                    d["q_end"] = [c[0] for c in cmds]
                    if not d.get("summary"):
                        d["summary"] = True
                        print("[duo] cycle done:")
                        for ph, tt, m in d["rows"]:
                            print(f"         {ph:15s} t {tt:5.2f} s  span {m['span']:+6.1f} mm  rot grip A {m['rot'][0]:+6.1f} "
                                  f"middle {m['rot'][1]:+6.1f} grip B {m['rot'][2]:+6.1f}  deform p95 {m['deform']:5.1f} mm")
                    if d.get("loop", DUO_LOOP):
                        d.update(planned=False, q_end=d["q_end"], summary=False)
                    else:
                        d["single_done"] = True
                lab = st.get("ui_status")
                if lab is not None:
                    held = [0 if g is None else len(g["idx"]) for g in st["grasps"]]
                    lab.text = (f"AUTO | {name} | A {held[0]} / B {held[1]} held | pull {_duo.pull * 1000:.0f} mm/hand | "
                                f"twist {_duo.twist_deg:.0f} deg")

            def _on_physx_step(dt, _key=_KEY, _step=_step, _start=_start, _reset=_reset):
                state = getattr(builtins, _key, None)
                if state is None or state["dead"]:
                    return
                try:
                    if state["phase"] == "stopping":
                        _reset(state)
                    if state["phase"] == "idle":
                        _start(state)
                    if abs(dt - 1.0 / 60.0) > 1e-4 and not state.get("dt_warned"):
                        state["dt_warned"] = True
                        print(f"[duo] warning: PhysX step is {dt * 1e3:.2f} ms; set Physics Scene to 60 Hz for one XPBD solve per step")
                    _step(state)
                except Exception as exc:  # noqa: BLE001
                    state["dead"] = True
                    import traceback
                    traceback.print_exc()
                    print(f"[duo] physics-step error: {type(exc).__name__}: {exc}")
            def _on_update(_event, _key=_KEY, _reset=_reset, _start=_start, _scale=_s,
                           _move_eps=RENDER_MOVE_EPS * _H, _write=_write_fabric, _read=_read_shape,
                           _gs_path=_splat_path, _rotation=_Rm, _translation=_tw):
                state = getattr(builtins, _key, None)
                if state is None:
                    return
                try:
                    timeline = omni.timeline.get_timeline_interface()
                    if not timeline.is_playing():
                        if timeline.is_stopped() and state["phase"] not in ("idle", "stopping"):
                            state["phase"], state["stop_wait"] = "stopping", 10
                        elif state["phase"] == "stopping":
                            state["stop_wait"] -= 1
                            if state["stop_wait"] <= 0:
                                _reset(state)
                                print("[duo] reset after timeline stop")
                        elif state["phase"] == "idle" and state["rewrite_in"] and state["fab"] is not None:
                            state["idle_frames"] += 1
                            if state["idle_frames"] >= state["rewrite_in"][0]:
                                state["rewrite_in"].pop(0)
                                _write(state, *state["fab"])
                        return
                    if state["dead"]:
                        return
                    if state["phase"] == "stopping":
                        _reset(state)
                    if state["phase"] == "idle":
                        _start(state)
                    if state["frames"] % 10 == 0:
                        selection = omni.usd.get_context().get_selection()
                        paths = selection.get_selected_prim_paths()
                        selected = [
                            path for path in paths
                            if path == _gs_path or path.startswith(_gs_path + "/") or _gs_path.startswith(path.rstrip("/") + "/")
                        ]
                        if selected:
                            selection.set_selected_prim_paths([path for path in paths if path not in selected], True)
                            print("[duo] deselected the deforming Gaussian prim during playback to protect the RTX renderer")
                    if KEEP_GRID and _settings.get(_GIZMO_KEY) is False:
                        _settings.set(_GIZMO_KEY, True)
                    steps = state["new_steps"]
                    if steps == 0:
                        return
                    state["new_steps"] = 0
                    t_sim = state["steps"] / 60.0
                    t_read = time.perf_counter()
                    positions, scales, rotations = _read(state["sim"])
                    center = _scale * (_rotation @ positions.astype(np.float64).mean(0)) + _translation
                    if state["com_prev"] is not None:
                        state["model_velocity"] = (center - state["com_prev"]) * 60.0 / steps
                    state["com_prev"] = center
                    if float(np.abs(positions - state["P_last"]).max()) > _move_eps:
                        _write(state, positions, scales, rotations)
                        state["P_last"] = positions
                        state["writes"] += 1
                    state["t_io"] += time.perf_counter() - t_read
                    state["frames"] += 1
                    _duo_step(state, positions, t_sim)
                    if state["steps"] >= state["next_log"]:
                        state["next_log"] = state["steps"] + 60
                        _enabled, up, height, _gravity = state["ground"]
                        heights = positions.astype(np.float64) @ np.asarray(up) - height
                        wall = time.perf_counter() - state["wall0"]
                        solved = state["steps"] - state["steps0"]
                        rendered = max(state["frames"], 1)
                        contacts = int(np.sum(state["col_hits"])) if state.get("col_hits") is not None else 0
                        print(
                            f"[duo] t {t_sim:5.1f} s | mean height {_scale * heights.mean():.3f} m | "
                            f"lowest {_scale * np.percentile(heights, 0.2):.3f} m | {solved / rendered:.1f} steps/frame | "
                            f"XPBD {state['t_step'] / max(solved, 1) * 1e3:.1f} ms/step | "
                            f"readback + Fabric {state['t_io'] / rendered * 1e3:.1f} ms/frame | "
                            f"renderer updates {state['writes']}/{rendered} | contacts {contacts} | "
                            f"{rendered / max(wall, 1e-9):.1f} FPS"
                        )
                        state["frames"] = 0
                        state["t_step"] = 0.0
                        state["t_io"] = 0.0
                        state["writes"] = 0
                        state["wall0"] = time.perf_counter()
                        state["steps0"] = state["steps"]
                except Exception as exc:  # noqa: BLE001
                    state["sub"] = None
                    state["step_sub"] = None
                    print(f"[duo] update error: {type(exc).__name__}: {exc}")
            # Subscribe after all callbacks have captured their local state.
            _st["step_sub"] = omni.physx.get_physx_interface().subscribe_physics_on_step_events(_on_physx_step, False, 0)
            _st["sub"] = carb.eventdispatcher.get_eventdispatcher().observe_event(
                event_name=omni.kit.app.GLOBAL_EVENT_UPDATE,
                on_event=_on_update,
                observer_name="gs_duo_runtime",
            )
            setattr(builtins, _KEY, _st)

            try:
                import omni.ui as ui

                def _duo_click(request, _key=_KEY):
                    state = getattr(builtins, _key, None)
                    if state is None:
                        return
                    if request == "manual":
                        for row in _offset_models:
                            for model in row:
                                model.set_value(0.0)
                    config = {
                        "pull": _pull_model.get_value_as_float() / 1000.0,
                        "lift": _lift_model.get_value_as_float() / 1000.0,
                        "twist": _twist_model.get_value_as_float(),
                        "speed": _speed_model.get_value_as_float(),
                        "offsets": [[model.get_value_as_float() / 1000.0 for model in row] for row in _offset_models],
                    }
                    state["duo"]["ui_request"] = (request, config)

                def _slider(label, value, low, high, step):
                    with ui.HStack(height=24, spacing=5):
                        ui.Label(label, width=150)
                        model = ui.SimpleFloatModel(float(value))
                        ui.FloatSlider(model=model, min=float(low), max=float(high), step=float(step))
                    return model

                _offset_models = []
                _window = ui.Window("Two-arm 3DGS control", width=470, height=700)
                with _window.frame:
                    with ui.VStack(spacing=5):
                        ui.Label(
                            "Press Play and wait for the model to settle. Auto pulls both ends before twisting about the grip axis. "
                            "Manual exposes independent arm offsets, grasping, pull, and twist controls.",
                            word_wrap=True,
                            height=50,
                        )
                        _st["ui_status"] = ui.Label("Waiting for Play", height=22)
                        with ui.HStack(height=28, spacing=4):
                            ui.Button("Auto loop", clicked_fn=lambda: _duo_click("auto_loop"))
                            ui.Button("Run once", clicked_fn=lambda: _duo_click("run_once"))
                            ui.Button("Manual", clicked_fn=lambda: _duo_click("manual"))
                        ui.Separator(height=8)
                        ui.Label("Automatic and paired manual action settings", height=20)
                        _pull_model = _slider("Pull per hand [mm]", DUO_PULL_M * 1000.0, 0.0, 250.0, 1.0)
                        _lift_model = _slider("Lift [mm]", DUO_PULL_UP_M * 1000.0, 0.0, 120.0, 1.0)
                        _twist_model = _slider("Twist each way [deg]", DUO_TWIST_DEG, 0.0, 180.0, 1.0)
                        _speed_model = _slider("Move time [s]", DUO_MOVE_S, 2.0, 8.0, 0.25)
                        ui.Separator(height=8)
                        ui.Label("TCP offsets from the pose active when Manual was selected [mm]", word_wrap=True, height=30)
                        for label in ("Arm A", "Arm B"):
                            with ui.HStack(height=23, spacing=5):
                                ui.Label(label, width=45)
                                row = []
                                for axis in "XYZ":
                                    model = ui.SimpleFloatModel(0.0)
                                    ui.Label(axis, width=12)
                                    ui.FloatSlider(model=model, min=-250.0, max=250.0, step=1.0)
                                    row.append(model)
                                _offset_models.append(row)
                        with ui.HStack(height=28, spacing=4):
                            ui.Button("Move both", clicked_fn=lambda: _duo_click("move_both"))
                            ui.Button("Move A", clicked_fn=lambda: _duo_click("move_a"))
                            ui.Button("Move B", clicked_fn=lambda: _duo_click("move_b"))
                        ui.Separator(height=8)
                        ui.Button("Approach + grasp both", clicked_fn=lambda: _duo_click("approach_grab"))
                        with ui.HStack(height=28, spacing=4):
                            ui.Button("Grasp A", clicked_fn=lambda: _duo_click("grab_a"))
                            ui.Button("Grasp B", clicked_fn=lambda: _duo_click("grab_b"))
                            ui.Button("Grasp both", clicked_fn=lambda: _duo_click("grab_both"))
                            ui.Button("Release both", clicked_fn=lambda: _duo_click("release"))
                        with ui.HStack(height=28, spacing=4):
                            ui.Button("Pull", clicked_fn=lambda: _duo_click("pull"))
                            ui.Button("Pull then twist", clicked_fn=lambda: _duo_click("pull_twist"))
                            ui.Button("Set pull + twist", clicked_fn=lambda: _duo_click("twist"))
                        with ui.HStack(height=28, spacing=4):
                            ui.Button("Return to grip", clicked_fn=lambda: _duo_click("reset_grip"))
                            ui.Button("Relax", clicked_fn=lambda: _duo_click("relax"))
                _st["ui_window"] = _window
                print("[duo] two-arm control window ready")
            except Exception as exc:  # noqa: BLE001
                print(f"[duo] no control window: {exc}")

            print(
                f"[duo] ready in {time.perf_counter() - _t0:.1f} s: {_name}, {len(_P0):,} Gaussians, scale {_model_scale:g}, "
                f"height {_s * _H:.2f} m, ground z {_z_floor:.3f}, gravity {_st['ground'][3] * _s:.2f} m/s^2, "
                f"{SUBSTEPS} physics steps per rendered frame. Press Play; run this script again to stop and restore the scene."
            )
