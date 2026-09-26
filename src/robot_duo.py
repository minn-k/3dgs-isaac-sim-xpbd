"""
두 Franka가 물체 양 끝을 잡고 당기기(stretch)·비틀기(twist)를 수행하는 runtime_loop.py의 duo 모드.

  팔 A (뒤, -X) : 물체의 가장 높은 곳(머리 정수리)을 위에서 잡는다.
  팔 B (앞, +X) : 바닥 가까이에서 B 쪽으로 가장 튀어나온 곳(발)을 위에서 잡는다.
  stretch : 두 손이 각자 자기 쪽으로 크게 당긴다 (A 는 뒤·위, B 는 앞·위) → 버틴다 → 돌아온다.
  twist   : (당긴 채로) 두 손이 두 손을 잇는 선 둘레로 반대 방향으로 천천히 돈다 (빨래 짜기) → 버틴다 → 되돌린다.
  잡기·놓기는 runtime_loop.py의 구간 이름으로 처리한다: 'close'가 끝나면 두 손 모두 잡고, 'open'에 들어가면 놓는다.
수동 조종(HandServo): 손마다 목표 자세로 속도 제한을 두고 따라간다 — 조종 창의 당기기·들기·비틀기 슬라이더가 목표를 정한다.
"""
import numpy as np

from robot_arm import approach_frame
from robot_common import FINGER_OPEN, _hand_frame, _polyline_at, _slerp_R, _smooth, grip_opening


def rot_axis(axis, deg):
    """축 axis 둘레 deg 도 회전 (Rodrigues)."""
    k = np.asarray(axis, float)
    k = k / np.linalg.norm(k)
    a = np.radians(deg)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def _local_top(W, xy, r=0.015):
    """xy 둘레 반경 r 안 가우시안의 가장 높은 z (윗면)."""
    m = np.linalg.norm(W[:, :2] - xy[:2], axis=1) < r
    return float(W[m, 2].max()) if m.any() else float(W[:, 2].max())


class _ArmPath:
    """팔 하나의 구간 목록 (이름, 길이 s, 끝 손끝 위치, 끝 손 자세, 손가락 벌림) → 시각 t 의 목표 → IK."""

    def __init__(self, arm, q_now):
        self.arm = arm
        self.q = np.array(q_now, float)
        p, R = arm.tcp(self.q)
        self.start = (p, R, float(arm.finger_cmd))
        self.segments = []

    def add(self, name, dur, p, R, finger):
        self.segments.append((name, float(dur), np.asarray(p, float), np.asarray(R, float), float(finger)))

    def target(self, t):
        p0, R0, f0 = self.start
        for name, d, p, R, fin in self.segments:
            if t < d:
                s = _smooth(t / d)
                pt = _polyline_at(p0, p, s) if p.ndim == 2 else p0 + (p - p0) * s
                return pt, _slerp_R(R0, R, s), f0 + (fin - f0) * s, name
            t -= d
            p0, R0, f0 = (p[-1] if p.ndim == 2 else p), R, fin
        name, _d, p, R, fin = self.segments[-1]
        return (p[-1] if p.ndim == 2 else p), R, fin, "done"

    def command(self, t, iters=6):
        p, R, fin, name = self.target(t)
        self.q, ep, _er = self.arm.ik(p, R, self.q, iters=iters)
        return self.q, fin, name, p, ep


class DuoDemo:
    """두 팔의 같은 이름·같은 길이 구간. plan() 뒤 command(t) → [(q, 손가락, 구간 이름, 손끝 목표, IK 오차)] × 2."""

    def __init__(self, arms, task="both", grasp_depth=0.03, squeeze=0.008, pad_r=0.0175, hover=0.10,
                 pull=0.12, pull_up=0.05, twist_deg=90.0, twist_axis="grip", twist_step_deg=30.0, speed_s=4.0,
                 floor_z=0.0):
        self.arms = arms
        self.task, self.grasp_depth, self.squeeze, self.pad_r, self.hover = task, grasp_depth, squeeze, pad_r, hover
        self.pull, self.pull_up, self.twist_deg, self.twist_axis = pull, pull_up, twist_deg, twist_axis
        self.twist_step, self.speed_s, self.floor_z = twist_step_deg, speed_s, floor_z
        self.paths, self.info = [], {}

    def pick_points(self, W):
        """A: 가장 높은 곳 둘레(위 6 cm 의 가운데 xy), B: 몸 높이 35% 아래에서 B 받침 쪽으로 가장 튀어나온 점의 3 cm 둘레.
        그래프에서 분리된 부유 가우시안은 제외한다. 중앙값에서의 거리가
        99 백분위의 2 배를 넘는 점."""
        r = np.linalg.norm(W - np.median(W, axis=0), axis=1)
        W = W[r < 2.0 * np.percentile(r, 99)]
        zmin, zmax = float(W[:, 2].min()), float(W[:, 2].max())
        top = W[W[:, 2] > zmax - 0.06]
        a_xy = np.median(top[:, :2], axis=0)
        bA, bB = self.arms[0].base[:3, 3], self.arms[1].base[:3, 3]
        d = (bB - bA)[:2]
        d /= np.linalg.norm(d)
        low = W[W[:, 2] < zmin + 0.35 * (zmax - zmin)]
        tip = low[np.argmax(low[:, :2] @ d)]
        near = low[np.linalg.norm(low[:, :2] - tip[:2], axis=1) < 0.03]
        b_xy = near[:, :2].mean(0)
        a = np.r_[a_xy, _local_top(W, a_xy)]
        b = np.r_[b_xy, _local_top(W, b_xy)]
        return a, b, np.r_[d, 0.0]

    def grasp_poses(self, W):
        """두 손의 잡는 자세: 손끝 위치 2 개, 손 자세 (위에서 아래로, 손가락은 당기는 방향에 가로로), 손가락 벌림, 당기는 방향."""
        a_top, b_top, d = self.pick_points(W)
        up = np.array([0.0, 0.0, 1.0])
        R0 = approach_frame([0.0, 0.0, -1.0], np.cross(up, d))
        grasps, grips = [], []
        for top in (a_top, b_top):
            g = top - np.array([0.0, 0.0, self.grasp_depth])
            g[2] = max(g[2], self.floor_z + 0.012)             # 손끝이 바닥에 닿지 않게
            grasps.append(g)
            grips.append(grip_opening(W, _hand_frame(g, R0), FINGER_OPEN, self.pad_r, self.squeeze))
        return {"grasp": grasps, "grip": grips, "R0": R0, "pull_dir": [-d, d], "top": [a_top, b_top], "dir": d}

    def pose_at(self, gp, pull, lift, twist):
        """잡는 자세 gp 에서 당기기(각자 자기 쪽으로) pull·들기 lift·비틀기 twist [deg] 만큼 옮긴 두 손의 (위치, 자세).
        비틀기 축: 'grip' = 옮긴 두 손끝을 잇는 선 (빨래 짜기), 'vertical' = 연직축. A 는 +, B 는 − 로 돈다."""
        up = np.array([0.0, 0.0, 1.0])
        ps = [g + dr * pull + up * lift for g, dr in zip(gp["grasp"], gp["pull_dir"])]
        axis = ps[1] - ps[0] if self.twist_axis == "grip" else up
        base_R = gp["R0"]
        base_Rs = [base_R, base_R] if np.asarray(base_R).shape == (3, 3) else base_R
        Rs = [rot_axis(axis, sgn * twist) @ base_Rs[k] for k, sgn in enumerate((1.0, -1.0))]
        return ps, Rs

    def reach_limits(self, gp, q_now, pull, lift, twist, tol=0.008, rot_tol=0.05):
        """두 팔이 실제로 닿는 당기기·비틀기로 줄인다 (기구학만: 잡는 자세 → 당긴 자세 → 비틀기를 20° 씩, IK 를 이어
        풀어 손끝 오차 tol [m]·자세 오차 rot_tol 안). 안 닿으면 이분 탐색 (당기기 ~1 cm, 비틀기 ~8° 까지). → (pull, lift, twist)."""
        R0s = gp["R0"] if np.asarray(gp["R0"]).ndim == 3 else [gp["R0"], gp["R0"]]
        q_grip = [arm.ik(gp["grasp"][k], R0s[k], q_now[k], iters=120)[0] for k, arm in enumerate(self.arms)]

        def pulled(pull_):
            ps, Rs = self.pose_at(gp, pull_, lift, 0.0)
            qs = []
            for k, arm in enumerate(self.arms):
                q, e, er = arm.ik(ps[k], Rs[k], q_grip[k], iters=80)
                if e > tol or er > rot_tol:
                    return None
                qs.append(q)
            return qs

        def twisted(qs, pull_, twist_):
            n = int(np.ceil(abs(twist_) / 20.0))
            for k, arm in enumerate(self.arms):
                q = qs[k]
                for i in range(1, n + 1):
                    ps_t, Rs_t = self.pose_at(gp, pull_, lift, twist_ * i / n)
                    q, e, er = arm.ik(ps_t[k], Rs_t[k], q, iters=30)
                    if e > tol or er > rot_tol:
                        return False
            return True

        qs = pulled(pull)
        if qs is None:                                      # 당기기: 닿는 가장 큰 값 (이분 탐색)
            lo, hi, qs = 0.0, pull, pulled(0.0)
            for _ in range(5):
                mid = 0.5 * (lo + hi)
                qm = pulled(mid)
                if qm is not None:
                    lo, qs = mid, qm
                else:
                    hi = mid
            pull = lo
        if qs is None:
            return 0.0, lift, 0.0
        if abs(twist) > 1e-6 and not twisted(qs, pull, twist):   # 비틀기: 같은 방식
            lo, hi = 0.0, abs(twist)
            for _ in range(5):
                mid = 0.5 * (lo + hi)
                if twisted(qs, pull, np.sign(twist) * mid):
                    lo = mid
                else:
                    hi = mid
            twist = float(np.sign(twist) * lo)
        return pull, lift, twist

    def plan(self, W, q_now):
        """W: 자리 잡은 물체의 월드 좌표. q_now: 두 팔의 지금 관절."""
        gp = self.grasp_poses(W)
        up = np.array([0.0, 0.0, 1.0])
        self.paths = [_ArmPath(arm, q) for arm, q in zip(self.arms, q_now)]
        stretch = self.task in ("stretch", "both")
        twist = self.task in ("twist", "both")
        pull = self.pull if stretch else 0.0
        lift = self.pull_up if stretch else 0.005
        tw = self.twist_deg if twist else 0.0
        asked = (pull, lift, tw)
        pull, lift, tw = self.reach_limits(gp, q_now, pull, lift, tw)      # 슬라이더 값이 팔 밖이면 닿는 만큼만
        twist = twist and abs(tw) >= 1.0
        gp["used"] = {"pull": pull, "lift": lift, "twist": tw, "asked": asked}
        n = max(1, int(np.ceil(abs(tw) / self.twist_step)))

        def both(name, dur, pull_, lift_, twist_, finger):
            ps, Rs = self.pose_at(gp, pull_, lift_, twist_)
            for k, path in enumerate(self.paths):
                f = finger if finger is not None else gp["grip"][k][0]
                path.add(name, dur, ps[k], Rs[k], f)

        for k, path in enumerate(self.paths):
            g = gp["grasp"][k]
            path.add("move over", 2.5, g + up * self.hover, gp["R0"], FINGER_OPEN)
            path.add("descend", 1.5, g, gp["R0"], FINGER_OPEN)
            path.add("close", 0.8, g, gp["R0"], gp["grip"][k][0])
        both("settle", 0.5, 0.0, 0.0, 0.0, None)
        if stretch:
            both("pull", self.speed_s, pull, lift, 0.0, None)          # 각자 자기 쪽으로 크게 당긴다
            both("hold stretched", 1.5, pull, lift, 0.0, None)
        if twist:                                                  # 당긴 채로 비튼다 (한 구간 ≤ twist_step 도)
            for i in range(1, n + 1):
                both("twist", self.speed_s / n, pull, lift, tw * i / n, None)
            both("hold twisted", 1.5, pull, lift, tw, None)
            for i in range(n - 1, -1, -1):
                both("untwist", self.speed_s / n, pull, lift, tw * i / n, None)
        if stretch:
            both("relax", self.speed_s, 0.0, 0.005, 0.0, None)
        both("rest", 1.0, 0.0, 0.005, 0.0, None)
        both("open", 0.6, 0.0, 0.005, 0.0, FINGER_OPEN)
        both("retreat", 1.2, 0.0, 0.12, 0.0, FINGER_OPEN)
        both("watch", 2.0, 0.0, 0.12, 0.0, FINGER_OPEN)
        self.info = gp
        return gp

    @property
    def duration(self):
        return sum(s[1] for s in self.paths[0].segments) if self.paths else 0.0

    def command(self, t, iters=6):
        return [p.command(t, iters) for p in self.paths]


class HandServo:
    """수동 조종: 손 하나가 목표 (손끝 위치, 손 자세, 손가락) 로 속도 제한을 두고 따라간다. 경유점 목록도 받는다."""

    def __init__(self, arm, q, v_max=0.08, w_max_deg=45.0):
        self.arm, self.q = arm, np.array(q, float)
        self.p, self.R = arm.tcp(self.q)
        self.finger = float(arm.finger_cmd)
        self.v_max, self.w_max = v_max, np.radians(w_max_deg)
        self.queue = []                                   # [(p, R, finger)] — 앞에서부터 차례로

    def go(self, p, R, finger=None, via=()):
        self.queue = [(np.asarray(v[0], float), np.asarray(v[1], float), v[2]) for v in via]
        self.queue.append((np.asarray(p, float), np.asarray(R, float), finger))

    def arrived(self):
        return not self.queue

    def step(self, dt, iters=6):
        if self.queue:
            p_t, R_t, f_t = self.queue[0]
            dp = p_t - self.p
            dist = float(np.linalg.norm(dp))
            Rr = self.R.T @ R_t
            ang = float(np.arccos(np.clip((np.trace(Rr) - 1.0) / 2.0, -1.0, 1.0)))
            s = min(1.0, self.v_max * dt / max(dist, 1e-9), self.w_max * dt / max(ang, 1e-9))
            self.p = self.p + dp * s
            self.R = _slerp_R(self.R, R_t, s)
            if f_t is not None:
                self.finger += float(np.clip(f_t - self.finger, -0.05 * dt, 0.05 * dt))
            if s >= 1.0 and (f_t is None or abs(self.finger - f_t) < 1e-4):
                self.queue.pop(0)
        self.q, ep, _er = self.arm.ik(self.p, self.R, self.q, iters=iters)
        return self.q, self.finger, self.p, ep
