"""Phase 3: local path planner with safe stop.

Pure module (no CARLA). Works in the road frame (s, d) of the ego-lane reference
line. ASSUMPTION: the reference line is straight or nearly so (the CARLA adapter
rejects segments bending > 2 deg), so a rectangle in the map frame stays a
rectangle in (s, d) and road-frame curvature ~ d''(s).

Method (sampled lateral-offset candidates):
  * Each candidate moves the ego origin from its current (d0, slope m0) to a
    target offset d1 with a quintic over length L1 (zero end slope/curvature),
    optionally starting after a delay, holding d1, and optionally returning to
    d_home after the last cone ahead.
  * A candidate is SAFE if, at every sample along the horizon, the full ego
    rectangle (bbox + buffer) (a) does not intersect any cone footprint inflated
    by the margin and (b) has all four corners inside the union of permitted
    corridors; and its curvature stays below the limit.
  * PASS: cheapest safe candidate (deviation from d_home + curvature + change
    from the previous target, for hysteresis).
  * STOP: no safe candidate. The keep-lane path is driven up to its first
    conflict minus stop_buffer, with a speed profile v = sqrt(2 a (s_stop - s)).

Physical free space and permissibility are checked separately: (a) vs (b).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .frames import wrap_angle
from .observations import ConeObservation
from .road import RoadGeometry
from .scenario import Region
from .state import EgoState, VehicleGeometry


@dataclass(frozen=True)
class PlannerConfig:
    horizon_m: float = 45.0
    sample_spacing_m: float = 0.5
    lateral_step_m: float = 0.15
    transition_lengths_m: tuple = (12.0, 18.0, 25.0)
    transition_delays_m: tuple = (0.0, 4.0, 8.0)
    return_gap_m: float = 3.0
    cone_margin_m: float = 0.40
    vehicle_buffer_m: float = 0.05
    stop_buffer_m: float = 1.5
    max_curvature: float = 0.10
    max_lat_accel: float = 1.0
    target_speed_mps: float = 3.0
    plan_decel_mps2: float = 1.5
    max_decel_mps2: float = 3.0
    d_home: float = 0.0
    w_deviation: float = 1.0
    w_curvature: float = 20.0
    w_change: float = 0.1
    w_consistency: float = 5.0
    switch_margin: float = 0.3

    @classmethod
    def from_config(cls, cfg: dict, margin_m: float | None = None) -> "PlannerConfig":
        p, v = cfg.get("planner", {}), cfg.get("vehicle", {})
        return cls(
            horizon_m=p.get("horizon_m", 45.0), sample_spacing_m=p.get("sample_spacing_m", 0.5),
            lateral_step_m=p.get("lateral_step_m", 0.15),
            transition_lengths_m=tuple(p.get("transition_lengths_m", (12.0, 18.0, 25.0))),
            transition_delays_m=tuple(p.get("transition_delays_m", (0.0, 4.0, 8.0))),
            return_gap_m=p.get("return_gap_m", 3.0),
            cone_margin_m=p.get("base_clearance_m", 0.40) if margin_m is None else margin_m,
            vehicle_buffer_m=p.get("vehicle_buffer_m", 0.05), stop_buffer_m=p.get("stop_buffer_m", 1.5),
            max_curvature=p.get("max_curvature_1pm", 0.10), max_lat_accel=p.get("max_lat_accel_mps2", 1.0),
            target_speed_mps=v.get("target_speed_mps", 3.0), plan_decel_mps2=p.get("plan_decel_mps2", 1.5),
            max_decel_mps2=p.get("max_decel_mps2", 3.0), d_home=p.get("d_home_m", 0.0),
            w_deviation=p.get("w_deviation", 1.0), w_curvature=p.get("w_curvature", 20.0),
            w_change=p.get("w_change", 0.5), w_consistency=p.get("w_consistency", 5.0),
            switch_margin=p.get("switch_margin", 0.3),
        )


@dataclass
class ReferencePoint:
    x: float
    y: float
    yaw: float
    speed_mps: float
    time_s: float


@dataclass
class Plan:
    """Reference for the ACTOR ORIGIN in the map frame, distance-indexed by s."""
    decision: Literal["PASS", "STOP"]
    reason: str
    s: np.ndarray            # road-frame samples
    d: np.ndarray
    x: np.ndarray            # map frame
    y: np.ndarray
    yaw: np.ndarray
    v: np.ndarray            # reference speed at each sample
    kappa: np.ndarray        # approx. path curvature (1/m)
    d_target: float
    s_stop: float | None = None
    required_decel_mps2: float = 0.0
    n_candidates: int = 0
    n_safe: int = 0
    dd: np.ndarray | None = None     # d'(s)
    ddd: np.ndarray | None = None    # d''(s)
    info: dict = field(default_factory=dict)

    def reference_points(self) -> list[ReferencePoint]:
        ds = np.hypot(np.diff(self.x), np.diff(self.y))
        v_mid = 0.5 * (self.v[1:] + self.v[:-1])
        dt = np.where(v_mid > 1e-3, ds / np.maximum(v_mid, 1e-3), np.inf)
        t = np.concatenate([[0.0], np.cumsum(dt)])
        return [ReferencePoint(float(a), float(b), float(c), float(d), float(e))
                for a, b, c, d, e in zip(self.x, self.y, self.yaw, self.v, t)]


# ----------------------------------------------------------------- geometry

def quintic_blend(s: np.ndarray, s_start: float, length: float, d_a: float, d_b: float,
                  slope_a: float = 0.0):
    """d(s) from d_a (slope slope_a, zero curvature) to d_b (zero slope/curv.)
    over [s_start, s_start+length]; constant outside. Returns d, d', d''."""
    L = length
    # Hermite quintic in tau with d(0)=0, d'(0)=m*L, d''(0)=0, d(1)=1, d'(1)=d''(1)=0
    # applied to the normalised offset; slope handled additively.
    tau = np.clip((s - s_start) / L, 0.0, 1.0)
    inside = (s >= s_start) & (s <= s_start + L)
    delta = d_b - d_a
    m = slope_a * L                                  # slope in tau units
    # Basis: h(tau) = 10t^3-15t^4+6t^5 (0->1); g(tau) = t-6t^3+8t^4-3t^5 (slope 1 at 0, all else 0)
    h = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    g = tau - 6 * tau**3 + 8 * tau**4 - 3 * tau**5
    dh = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / L
    dg = (1 - 18 * tau**2 + 32 * tau**3 - 15 * tau**4) / L
    ddh = (60 * tau - 180 * tau**2 + 120 * tau**3) / L**2
    ddg = (-36 * tau + 96 * tau**2 - 60 * tau**3) / L**2
    d = d_a + delta * h + m * g
    d1 = np.where(inside, delta * dh + m * dg, 0.0)
    d2 = np.where(inside, delta * ddh + m * ddg, 0.0)
    before = s < s_start
    d = np.where(before, d_a + slope_a * (s - s_start), d)   # straight-line extension before start
    d1 = np.where(before, slope_a, d1)
    return d, d1, d2


def _rects(cs, cd, psi, half_l, half_w):
    """Corners (..., 4, 2) of rectangles in the road frame."""
    c, s = np.cos(psi), np.sin(psi)
    loc = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=float)
    lx = loc[:, 0] * half_l
    ly = loc[:, 1] * half_w
    xs = cs[..., None] + c[..., None] * lx - s[..., None] * ly
    ys = cd[..., None] + s[..., None] * lx + c[..., None] * ly
    return np.stack([xs, ys], axis=-1)


def _sat_intersect(ego, cones):
    """Separating-axis test. ego: (C, n, 4, 2); cones: (M, 4, 2).
    Returns bool (C, n, M): True where rectangles overlap."""
    if cones.shape[0] == 0:
        return np.zeros(ego.shape[:2] + (0,), dtype=bool)
    e_ax = np.stack([ego[..., 0, :] - ego[..., 1, :], ego[..., 1, :] - ego[..., 2, :]], axis=-2)   # (C,n,2,2)
    k_ax = np.stack([cones[:, 0] - cones[:, 1], cones[:, 1] - cones[:, 2]], axis=1)              # (M,2,2)
    E = ego[:, :, None]                       # (C,n,1,4,2)
    K = cones[None, None]                     # (1,1,M,4,2)
    sep = np.zeros(ego.shape[:2] + (cones.shape[0],), dtype=bool)
    axes = [e_ax[:, :, None, 0], e_ax[:, :, None, 1],
            np.broadcast_to(k_ax[None, None, :, 0], e_ax.shape[:2] + (cones.shape[0], 2)),
            np.broadcast_to(k_ax[None, None, :, 1], e_ax.shape[:2] + (cones.shape[0], 2))]
    for ax in axes:                           # ax: (C,n,M,2) or (C,n,1,2)
        pe = np.einsum("...kj,...j->...k", E, ax)
        pk = np.einsum("...kj,...j->...k", K, ax)
        sep |= (pe.max(-1) < pk.min(-1)) | (pk.max(-1) < pe.min(-1))
    return ~sep


def _outside_corridors(cs, cd, corridors: list[Region]):
    """Bool mask: point (s, d) not inside any permitted corridor. Points beyond
    the corridors' total s-extent are clamped into it (behind the start / past
    the end of the scenario nothing is defined, so the nearest rule applies)."""
    s_lo = min(c.s_min for c in corridors)
    s_hi = max(c.s_max for c in corridors)
    cs = np.clip(cs, s_lo, s_hi)
    inside = np.zeros(cs.shape, dtype=bool)
    for c in corridors:
        inside |= (cs >= c.s_min) & (cs <= c.s_max) & (cd >= c.d_min - 1e-9) & (cd <= c.d_max + 1e-9)
    return ~inside


def cone_rects_road(cones: list[ConeObservation], road: RoadGeometry, margin: float) -> np.ndarray:
    if not cones:
        return np.zeros((0, 4, 2))
    out = []
    for c in cones:
        fp = c.footprint(margin)
        s, d = road.reference.to_frenet(fp[:, 0], fp[:, 1])
        out.append(np.column_stack([s, d]))
    return np.array(out)


# ------------------------------------------------------------------ planner

class LocalPlanner:
    def __init__(self, cfg: PlannerConfig, geom: VehicleGeometry):
        self.cfg = cfg
        self.geom = geom
        self.prev: Plan | None = None

    def plan(self, ego: EgoState, cones: list[ConeObservation], road: RoadGeometry,
             corridors: list[Region]) -> Plan:
        cfg, g = self.cfg, self.geom
        ref = road.reference
        s0, d0 = ref.to_frenet(ego.x, ego.y)
        rel = float(wrap_angle(ego.yaw - float(ref.heading_at(s0))))
        m0 = math.tan(max(-0.5, min(0.5, rel)))
        s_route_end = min(ref.length, max(c.s_max for c in corridors)) - g.length_m / 2 - 0.5
        H = max(0.0, min(cfg.horizon_m, s_route_end - s0))
        S = s0 + np.arange(0.0, H + 1e-9, cfg.sample_spacing_m)
        if len(S) < 2:
            S = np.array([s0, s0 + cfg.sample_spacing_m])

        K = cone_rects_road(cones, road, cfg.cone_margin_m)
        if len(K):
            ks_lo, ks_hi = K[:, :, 0].min(1), K[:, :, 0].max(1)
            keep = (ks_hi >= s0 - g.length_m) & (ks_lo <= S[-1] + g.length_m)
            K, ks_hi = K[keep], ks_hi[keep]
        ahead_hi = ks_hi[ks_hi > s0].max() if len(K) and np.any(ks_hi > s0) else None

        # -------- candidate set
        half_w = g.width_m / 2 + cfg.vehicle_buffer_m
        d_lo = min(c.d_min for c in corridors) + half_w
        d_hi = max(c.d_max for c in corridors) - half_w
        targets = set(np.round(np.arange(d_lo, d_hi + 1e-9, cfg.lateral_step_m), 4).tolist())
        targets |= {cfg.d_home, round(float(np.clip(d0, d_lo, d_hi)), 4)}
        cands = []
        for d1 in sorted(targets):
            for L1 in cfg.transition_lengths_m:
                for delay in cfg.transition_delays_m:
                    cands.append((d1, L1, delay, False))
                    if ahead_hi is not None and abs(d1 - cfg.d_home) > 1e-6:
                        cands.append((d1, L1, delay, True))
        s_ret = None if ahead_hi is None else ahead_hi + g.length_m / 2 + cfg.return_gap_m

        D = np.empty((len(cands), len(S))); D1 = np.empty_like(D); D2 = np.empty_like(D)
        for i, (d1, L1, delay, ret) in enumerate(cands):
            # Delayed candidates hold the current offset (slope 0) until s0 + delay;
            # the MPC absorbs any small initial heading error.
            d, dd, ddd = quintic_blend(S, s0 + delay, L1, d0, d1, m0 if delay == 0 else 0.0)
            if ret:
                start = max(s_ret, s0 + delay + L1)
                L2 = max(L1, 15.0)
                r, rd, rdd = quintic_blend(S, start, L2, d1, cfg.d_home)
                after = S >= start
                d, dd, ddd = np.where(after, r, d), np.where(after, rd, dd), np.where(after, rdd, ddd)
            D[i], D1[i], D2[i] = d, dd, ddd

        # "Continue" candidate: the previously committed PASS path, re-sampled on
        # the new s-grid. Without it, every replan restarts the transition curve at
        # the current state and the (flat-starting) quintic never makes progress.
        prev = self.prev
        # Extending a path that ends mid-curve would create a kink, so only reuse
        # it if it is straight at its end or the new horizon stays inside it.
        if prev is not None and prev.decision == "PASS" and prev.dd is not None \
                and abs(d0 - float(np.interp(s0, prev.s, prev.d))) < 0.3 \
                and (abs(prev.dd[-1]) < 1e-3 or S[-1] <= prev.s[-1] + 1e-9):
            cands.append((prev.d_target, 0.0, 0.0, "continue"))
            D = np.vstack([D, np.interp(S, prev.s, prev.d)])
            D1 = np.vstack([D1, np.interp(S, prev.s, prev.dd, right=0.0)])
            D2 = np.vstack([D2, np.interp(S, prev.s, prev.ddd, right=0.0)])

        bad = self._first_bad(S, D, D1, D2, K, corridors)       # index of first conflict, len(S) if none
        safe = bad >= len(S)
        n_safe = int(safe.sum())

        if n_safe:
            # Linear (not squared) deviation so small residual offsets are not
            # "free" and the ego returns exactly to d_home when it is safe.
            cost = (cfg.w_deviation * np.mean(np.abs(D - cfg.d_home), axis=1)
                    + cfg.w_curvature * np.max(np.abs(D2), axis=1))
            if self.prev is not None:   # small target hysteresis; never anchors to an arbitrary start offset
                cost += cfg.w_change * np.abs(np.array([c[0] for c in cands]) - self.prev.d_target)
            # Consistency with the previously committed PASS path over the shared
            # s-range: stops the re-planner from postponing a manoeuvre forever
            # ("shift later" always looks cheaper) and from oscillating.
            if self.prev is not None and self.prev.decision == "PASS":
                overlap = (S >= self.prev.s[0]) & (S <= self.prev.s[-1])
                if overlap.any():
                    d_prev = np.interp(S[overlap], self.prev.s, self.prev.d)
                    cost += cfg.w_consistency * np.mean((D[:, overlap] - d_prev) ** 2, axis=1)
            cost[~safe] = np.inf
            i = int(np.argmin(cost))
            # Commitment: keep the previous path while it stays safe unless an
            # alternative is better by a clear margin. Re-anchored "start a bit
            # later" candidates otherwise win by tiny amounts every cycle and the
            # manoeuvre is postponed until it is no longer feasible.
            if cands[-1][3] == "continue" and safe[-1] and cost[-1] - cost[i] < cfg.switch_margin:
                i = len(cands) - 1
            kind = "continue previous" if cands[i][3] == "continue" else \
                f"L={cands[i][1]:.0f}, delay={cands[i][2]:.0f}, return={cands[i][3]}"
            plan = self._make_plan("PASS", f"safe path to d={cands[i][0]:+.2f} ({kind})",
                                   road, S, D[i], D1[i], D2[i], cands[i][0], ego, None)
        else:
            # Keep-lane path (target = clip(d_home), quickest smooth transition).
            keep = [i for i, c in enumerate(cands) if abs(c[0] - np.clip(cfg.d_home, d_lo, d_hi)) < 1e-6
                    and c[3] is False and c[2] == 0.0]
            i = max(keep, key=lambda j: bad[j]) if keep else int(np.argmax(bad))
            s_conflict = S[min(bad[i], len(S) - 1)]
            s_stop = max(s0, s_conflict - cfg.stop_buffer_m)
            plan = self._make_plan("STOP", f"no safe path in permitted corridors; first conflict at "
                                   f"s={s_conflict:.1f}", road, S, D[i], D1[i], D2[i], cands[i][0], ego, s_stop)
        plan.n_candidates, plan.n_safe = len(cands), n_safe
        self.prev = plan
        return plan

    def _first_bad(self, S, D, D1, D2, K, corridors):
        cfg, g = self.cfg, self.geom
        psi = np.arctan(D1)
        off = g.bbox_center_offset_x_m
        cs = S[None, :] + off * np.cos(psi)
        cd = D + off * np.sin(psi)
        ego = _rects(cs, cd, psi, g.length_m / 2 + cfg.vehicle_buffer_m, g.width_m / 2 + cfg.vehicle_buffer_m)
        bad_pt = np.abs(D2) > cfg.max_curvature
        bad_pt |= _outside_corridors(ego[..., 0], ego[..., 1], corridors).any(-1)
        if len(K):
            # Only samples whose ego body can reach a cone along s need the SAT test.
            reach = math.hypot(g.length_m / 2, g.width_m / 2) + abs(off) + cfg.vehicle_buffer_m
            near = ((S[:, None] + reach >= K[:, :, 0].min(1)[None]) &
                    (S[:, None] - reach <= K[:, :, 0].max(1)[None])).any(1)
            idx = np.flatnonzero(near)
            if len(idx):
                sub = ego[:, idx]
                for lo in range(0, len(D), 256):     # chunk to bound memory
                    bad_pt[lo:lo + 256, idx] |= _sat_intersect(sub[lo:lo + 256], K).any(-1)
        first = np.where(bad_pt.any(1), bad_pt.argmax(1), len(S))
        return first

    def _make_plan(self, decision, reason, road, S, d, d1, d2, d_target, ego, s_stop):
        cfg = self.cfg
        x, y, h = road.reference.to_world(S, d)
        yaw = h + np.arctan(d1)
        kappa = d2 / (1 + d1 ** 2) ** 1.5
        v = np.full(len(S), cfg.target_speed_mps)
        v = np.minimum(v, np.sqrt(cfg.max_lat_accel / np.maximum(np.abs(kappa), 1e-6)))
        # Always able to stop by the end of the checked horizon (also the route end).
        v = np.minimum(v, np.sqrt(2 * cfg.plan_decel_mps2 * np.maximum(S[-1] - S, 0.0)))
        req = 0.0
        if decision == "STOP":
            dist = max(s_stop - S[0], 0.0)
            v0 = max(ego.speed_mps, 0.0)
            req = v0 ** 2 / (2 * dist) if dist > 1e-3 else (math.inf if v0 > 0.05 else 0.0)
            a = min(max(cfg.plan_decel_mps2, req), cfg.max_decel_mps2)
            v = np.minimum(v, np.sqrt(2 * a * np.maximum(s_stop - S, 0.0)))
        return Plan(decision, reason, S, d, x, y, yaw, v, kappa, float(d_target), s_stop, float(req),
                    dd=np.asarray(d1, float), ddd=np.asarray(d2, float))
