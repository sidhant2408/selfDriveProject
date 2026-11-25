"""Phase 4: nonlinear MPC on a kinematic bicycle model (CasADi + IPOPT).

State  z = [x, y, psi, v]  of the REAR AXLE in the map frame.
Input  u = [a, delta]      longitudinal acceleration (m/s^2), front-wheel angle (rad,
                           left positive in the map frame).
Model (forward Euler, valid at low speed):
    x+   = x + v cos(psi) dt
    y+   = y + v sin(psi) dt
    psi+ = psi + v/L tan(delta) dt
    v+   = v + a dt
Cost (p_k = actor ORIGIN = rear axle + offset * heading; the plan is for the origin):
       sum_k  w_pos |p_k - p_ref|^2 + w_yaw * 2(1 - cos(psi_k - psi_ref))
            + w_v (v_k - v_ref)^2 + w_accel a^2 + w_steer delta^2
            + w_daccel (a_k - a_{k-1})^2 + w_dsteer (delta_k - delta_{k-1})^2
       + w_terminal * (position + yaw + speed terms at k = N).
2(1 - cos e) ~ e^2 for small e and has no +-pi wrap discontinuity.
Hard constraints: |delta| <= max_steer, |delta_k - delta_{k-1}| <= rate*dt
(including against the previously applied delta), -max_brake <= a <= max_accel,
0 <= v <= max_speed.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import casadi as ca
import numpy as np

from .frames import wrap_angle
from .planner import Plan
from .state import EgoState


@dataclass(frozen=True)
class MPCConfig:
    horizon_steps: int = 20
    dt: float = 0.10
    wheelbase_m: float = 2.875
    rear_axle_offset_m: float = 1.44
    max_accel: float = 1.5
    max_brake: float = 3.0
    max_speed: float = 4.0
    max_steer: float = 0.45
    max_steer_rate: float = 0.6
    w_pos: float = 2.0
    w_yaw: float = 2.0
    w_v: float = 1.0
    w_accel: float = 0.1
    w_steer: float = 0.5
    w_daccel: float = 0.5
    w_dsteer: float = 10.0
    w_terminal: float = 5.0
    ipopt_max_iter: int = 100

    @classmethod
    def from_config(cls, cfg: dict) -> "MPCConfig":
        m, v = cfg.get("mpc", {}), cfg.get("vehicle", {})
        keys = {f: m[f] for f in ("horizon_steps", "dt", "w_pos", "w_yaw", "w_v", "w_accel", "w_steer",
                                  "w_daccel", "w_dsteer", "w_terminal", "ipopt_max_iter") if f in m}
        return cls(wheelbase_m=v.get("wheelbase_m", 2.875), rear_axle_offset_m=v.get("rear_axle_offset_m", 1.44),
                   max_accel=m.get("max_accel_mps2", 1.5), max_brake=m.get("max_brake_mps2", 3.0),
                   max_speed=m.get("max_speed_mps", 4.0), max_steer=m.get("max_steer_rad", 0.45),
                   max_steer_rate=m.get("max_steer_rate_rps", 0.6), **keys)


@dataclass
class MPCResult:
    ok: bool
    accel: float
    delta: float
    v_next: float               # predicted speed after one control step
    predicted: np.ndarray       # (N+1, 4) rear-axle states
    reference: np.ndarray       # (N+1, 4)
    solve_time_s: float
    status: str
    fallback: bool = False


def origin_to_rear(x, y, yaw, offset):
    return x - offset * np.cos(yaw), y - offset * np.sin(yaw)


def build_reference(plan: Plan, ego: EgoState, cfg: MPCConfig) -> np.ndarray:
    """Time-sample the plan for the horizon: start at the ego's projection on the
    path and advance along it at the plan's reference speed. Returns (N+1, 4)
    ACTOR-ORIGIN [x, y, psi, v] (the MPC cost evaluates the origin from its
    rear-axle state). A stop profile (v -> 0) makes the samples pile up at the
    stop point, which is exactly the target to hold."""
    px, py = plan.x, plan.y
    seg = np.hypot(np.diff(px), np.diff(py))
    p = np.concatenate([[0.0], np.cumsum(seg)])
    ex, ey = ego.x, ego.y
    # projection of the ego rear axle onto the path polyline
    a = np.column_stack([px[:-1], py[:-1]])
    u = np.column_stack([np.diff(px), np.diff(py)]) / np.maximum(seg, 1e-9)[:, None]
    rel = np.array([ex, ey]) - a
    t = np.clip(np.einsum("kj,kj->k", rel, u), 0.0, seg)
    dist = np.hypot(*(a + t[:, None] * u - np.array([ex, ey])).T)
    k = int(np.argmin(dist))
    pk = p[k] + t[k]
    yaw_u = np.unwrap(plan.yaw)
    out = np.empty((cfg.horizon_steps + 1, 4))
    for i in range(cfg.horizon_steps + 1):
        pk = min(pk, p[-1])
        v_ref = float(np.interp(pk, p, plan.v))
        out[i] = [np.interp(pk, p, px), np.interp(pk, p, py), np.interp(pk, p, yaw_u), v_ref]
        pk += v_ref * cfg.dt
    # keep reference yaw within +-pi of the current heading (cost is wrap-free anyway)
    out[:, 2] = ego.yaw + wrap_angle(out[:, 2] - ego.yaw)
    return out


class KinematicMPC:
    def __init__(self, cfg: MPCConfig):
        self.cfg = cfg
        N, dt, L = cfg.horizon_steps, cfg.dt, cfg.wheelbase_m
        opti = ca.Opti()
        Z = opti.variable(4, N + 1)
        U = opti.variable(2, N)
        z0 = opti.parameter(4)
        R = opti.parameter(4, N + 1)
        u_prev = opti.parameter(2)

        off = cfg.rear_axle_offset_m
        cost = 0
        for k in range(N):
            x, y, psi, v = Z[0, k], Z[1, k], Z[2, k], Z[3, k]
            a, de = U[0, k], U[1, k]
            opti.subject_to(Z[:, k + 1] == ca.vertcat(x + v * ca.cos(psi) * dt,
                                                      y + v * ca.sin(psi) * dt,
                                                      psi + v / L * ca.tan(de) * dt,
                                                      v + a * dt))
            # position cost on the ACTOR ORIGIN (rear axle + offset along heading),
            # which is the point the plan is expressed for.
            ox, oy = x + off * ca.cos(psi), y + off * ca.sin(psi)
            cost += (cfg.w_pos * ((ox - R[0, k]) ** 2 + (oy - R[1, k]) ** 2)
                     + cfg.w_yaw * 2 * (1 - ca.cos(psi - R[2, k]))
                     + cfg.w_v * (v - R[3, k]) ** 2
                     + cfg.w_accel * a ** 2 + cfg.w_steer * de ** 2)
            prev = u_prev if k == 0 else U[:, k - 1]
            cost += cfg.w_daccel * (a - prev[0]) ** 2 + cfg.w_dsteer * (de - prev[1]) ** 2
            opti.subject_to(opti.bounded(-cfg.max_steer_rate * dt, de - prev[1], cfg.max_steer_rate * dt))
        psiN, vN = Z[2, N], Z[3, N]
        xN, yN = Z[0, N] + off * ca.cos(psiN), Z[1, N] + off * ca.sin(psiN)
        cost += cfg.w_terminal * (cfg.w_pos * ((xN - R[0, N]) ** 2 + (yN - R[1, N]) ** 2)
                                  + cfg.w_yaw * 2 * (1 - ca.cos(psiN - R[2, N]))
                                  + cfg.w_v * (vN - R[3, N]) ** 2)
        opti.minimize(cost)
        opti.subject_to(Z[:, 0] == z0)
        opti.subject_to(opti.bounded(-cfg.max_brake, U[0, :], cfg.max_accel))
        opti.subject_to(opti.bounded(-cfg.max_steer, U[1, :], cfg.max_steer))
        opti.subject_to(opti.bounded(0.0, Z[3, :], cfg.max_speed))
        opti.solver("ipopt", {"expand": True, "print_time": False},
                    {"print_level": 0, "sb": "yes", "max_iter": cfg.ipopt_max_iter,
                     "tol": 1e-4, "acceptable_tol": 1e-3})
        self.opti, self.Z, self.U, self.z0, self.R, self.u_prev = opti, Z, U, z0, R, u_prev
        self._warm_Z = None
        self._warm_U = None
        self.last_u = np.zeros(2)

    def reset(self):
        self._warm_Z = self._warm_U = None
        self.last_u = np.zeros(2)

    def solve(self, ego: EgoState, ref: np.ndarray) -> MPCResult:
        cfg = self.cfg
        N = cfg.horizon_steps
        xr, yr = origin_to_rear(ego.x, ego.y, ego.yaw, cfg.rear_axle_offset_m)
        z0 = np.array([xr, yr, ego.yaw, max(ego.speed_mps, 0.0)])
        z0[3] = min(z0[3], cfg.max_speed)
        if not (np.all(np.isfinite(z0)) and np.all(np.isfinite(ref))):
            return self.fallback(z0, ref, 0.0, "non-finite state or reference")
        opti = self.opti
        opti.set_value(self.z0, z0)
        opti.set_value(self.R, ref.T)
        opti.set_value(self.u_prev, self.last_u)
        if self._warm_Z is not None:
            Zw = np.hstack([self._warm_Z[:, 1:], self._warm_Z[:, -1:]])
            Zw[:, 0] = z0
            Uw = np.hstack([self._warm_U[:, 1:], self._warm_U[:, -1:]])
        else:
            Zw = np.repeat(z0[:, None], N + 1, axis=1)
            Uw = np.zeros((2, N))
        opti.set_initial(self.Z, Zw)
        opti.set_initial(self.U, Uw)
        t0 = time.perf_counter()
        try:
            sol = opti.solve()
            status = opti.stats().get("return_status", "?")
            Zs, Us = np.asarray(sol.value(self.Z)), np.asarray(sol.value(self.U))
            ok = bool(opti.stats().get("success", False)) and np.all(np.isfinite(Us))
        except RuntimeError:
            status = opti.stats().get("return_status", "exception")
            ok, Zs, Us = False, None, None
        dt_s = time.perf_counter() - t0
        if not ok:
            return self.fallback(z0, ref, dt_s, status)
        self._warm_Z, self._warm_U = Zs, Us
        self.last_u = Us[:, 0].copy()
        return MPCResult(True, float(Us[0, 0]), float(Us[1, 0]), float(Zs[3, 1]), Zs.T, ref, dt_s, status)

    def fallback(self, z0, ref, dt_s, status) -> MPCResult:
        """Solver failed: brake hard, hold the previous wheel angle, and drop the
        warm start so the next solve starts clean."""
        self._warm_Z = self._warm_U = None
        a = -self.cfg.max_brake
        delta = float(self.last_u[1])
        self.last_u = np.array([a, delta])
        pred = np.repeat(z0[None], self.cfg.horizon_steps + 1, axis=0)
        return MPCResult(False, a, delta, max(z0[3] + a * self.cfg.dt, 0.0), pred, ref, dt_s, status, True)


def bicycle_step(z: np.ndarray, u: np.ndarray, L: float, dt: float) -> np.ndarray:
    """Same discrete model as the MPC, for tests and the offline plant."""
    x, y, psi, v = z
    a, de = u
    return np.array([x + v * math.cos(psi) * dt, y + v * math.sin(psi) * dt,
                     psi + v / L * math.tan(de) * dt, max(0.0, v + a * dt)])
