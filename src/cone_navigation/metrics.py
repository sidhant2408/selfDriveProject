"""Episode metrics, evaluated against TRUE cone geometry (never the observations).

Definitions
-----------
true clearance : shapely distance between the ego bounding box (no buffer) and
    each TRUE cone footprint (no margin); the episode value is the minimum over
    time and cones. NEGATIVE means the footprints overlap; its magnitude is
    sqrt(overlap area), a size proxy, not a penetration depth.
corridor violation : how far (m, road frame) any ego bbox corner lies outside
    the union of permitted corridors; 0 = always inside.
cross-track error : distance from the ego actor origin to the active plan path.
"""
from __future__ import annotations

import math

import numpy as np
from shapely.geometry import LineString, Point, Polygon

from .observations import ConeObservation
from .road import RoadGeometry
from .scenario import Region
from .state import EgoState, VehicleGeometry, ego_footprint


def corridor_violation(corners_sd: np.ndarray, corridors: list[Region]) -> float:
    s_lo = min(c.s_min for c in corridors)
    s_hi = max(c.s_max for c in corridors)
    worst = 0.0
    for s, d in corners_sd:
        s = min(max(s, s_lo), s_hi)
        dist = min(math.hypot(max(c.s_min - s, 0, s - c.s_max), max(c.d_min - d, 0, d - c.d_max))
                   for c in corridors)
        worst = max(worst, dist)
    return worst


class EpisodeMetrics:
    def __init__(self, truth: list[ConeObservation], road: RoadGeometry, corridors: list[Region],
                 geom: VehicleGeometry):
        self.road, self.corridors, self.geom = road, corridors, geom
        self.truth_polys = [(c.cone_id, Polygon(c.footprint())) for c in truth]
        self.rows: list[dict] = []
        self.solve_times: list[float] = []
        self.plan_times: list[float] = []
        self.solver_failures = 0
        self.collision_events: list = []

    def clearance(self, ego: EgoState) -> tuple[float, str | None]:
        if not self.truth_polys:
            return math.inf, None
        poly = Polygon(ego_footprint(ego, self.geom))
        best, who = math.inf, None
        for cid, cp in self.truth_polys:
            dist = poly.distance(cp)
            if dist == 0.0 and poly.intersects(cp):
                dist = -math.sqrt(poly.intersection(cp).area)
            if dist < best:
                best, who = dist, cid
        return best, who

    def update(self, t: float, ego: EgoState, plan, mpc_res, cmd: dict, plan_time_s: float | None,
               fresh_solve: bool = True) -> dict:
        s, d = self.road.reference.to_frenet(ego.x, ego.y)
        clr, who = self.clearance(ego)
        fp = ego_footprint(ego, self.geom)
        cs, cd = self.road.reference.to_frenet(fp[:, 0], fp[:, 1])
        viol = corridor_violation(np.column_stack([cs, cd]), self.corridors)
        xte = LineString(np.column_stack([plan.x, plan.y])).distance(Point(ego.x, ego.y)) if len(plan.x) > 1 else 0.0
        if mpc_res is not None and fresh_solve:
            self.solve_times.append(mpc_res.solve_time_s)
        if plan_time_s is not None:
            self.plan_times.append(plan_time_s)
        row = {
            "t_s": round(t, 4), "frame": ego.frame, "x_map": ego.x, "y_map": ego.y, "yaw_map": ego.yaw,
            "s": s, "d": d, "speed_mps": ego.speed_mps,
            "v_ref": float(mpc_res.reference[0, 3]) if mpc_res is not None else float("nan"),
            "accel_cmd": mpc_res.accel if mpc_res else 0.0, "delta_cmd": mpc_res.delta if mpc_res else 0.0,
            "mpc_ok": bool(mpc_res.ok) if mpc_res else True,
            "mpc_ms": (mpc_res.solve_time_s * 1e3) if (mpc_res and fresh_solve) else float("nan"),
            "decision": plan.decision, "d_target": plan.d_target, "s_stop": plan.s_stop,
            "clearance_m": clr, "nearest_cone": who, "corridor_violation_m": viol, "cross_track_m": xte,
            **cmd,
        }
        self.rows.append(row)
        return row

    def _steer_rate(self) -> float:
        """Max |d delta / dt| between consecutive MPC solves (commands are held
        between solves, so per-tick differences would overstate the rate)."""
        fresh = [r for r in self.rows if r["mpc_ms"] == r["mpc_ms"]]       # not NaN
        if len(fresh) < 2:
            return 0.0
        return float(max(abs(b["delta_cmd"] - a["delta_cmd"]) / max(b["t_s"] - a["t_s"], 1e-6)
                         for a, b in zip(fresh, fresh[1:])))

    def summary(self, outcome: str, dt_tick: float) -> dict:
        r = self.rows
        arr = lambda k: np.array([x[k] for x in r], dtype=float)
        speed, xte, delta = arr("speed_mps"), arr("cross_track_m"), arr("delta_cmd")
        t = arr("t_s")
        acc_meas = np.diff(speed) / dt_tick if len(speed) > 1 else np.zeros(1)
        st = np.array(self.solve_times) if self.solve_times else np.zeros(1)
        clr = arr("clearance_m")
        finite = clr[np.isfinite(clr)]
        return {
            "outcome": outcome,
            "duration_s": float(t[-1]) if len(t) else 0.0,
            "final_s": float(r[-1]["s"]) if r else 0.0,
            "final_speed_mps": float(speed[-1]) if len(speed) else 0.0,
            "min_true_clearance_m": float(finite.min()) if len(finite) else None,
            "final_true_clearance_m": float(finite[-1]) if len(finite) else None,
            "max_corridor_violation_m": float(arr("corridor_violation_m").max()) if r else 0.0,
            "cross_track_rms_m": float(np.sqrt(np.mean(xte ** 2))) if len(xte) else 0.0,
            "cross_track_max_m": float(xte.max()) if len(xte) else 0.0,
            "max_speed_mps": float(speed.max()) if len(speed) else 0.0,
            "peak_decel_measured_mps2": float(-acc_meas.min()) if len(acc_meas) else 0.0,
            "steer_rate_cmd_max_rps": self._steer_rate(),
            "mpc_solves": len(self.solve_times),
            "mpc_failures": self.solver_failures,
            "mpc_ms_mean": float(st.mean() * 1e3), "mpc_ms_p95": float(np.percentile(st, 95) * 1e3),
            "plan_ms_mean": float(np.mean(self.plan_times) * 1e3) if self.plan_times else 0.0,
            "plan_ms_max": float(np.max(self.plan_times) * 1e3) if self.plan_times else 0.0,
            "stop_decisions": int(sum(1 for x in r if x["decision"] == "STOP")),
            "sim_collision_events": [str(e) for e in self.collision_events][:10],
        }
