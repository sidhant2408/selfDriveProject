"""Offline plant: kinematic bicycle with first-order actuator lag (no CARLA).

Lets the full planner + MPC loop run in tests. It is deliberately close to the
MPC's own model, so it validates integration, not robustness to CARLA dynamics.
"""
from __future__ import annotations

import math

import numpy as np

from .mpc import bicycle_step
from .state import EgoState


class KinematicPlant:
    name = "kinematic"

    def __init__(self, start: EgoState, wheelbase_m: float, rear_axle_offset_m: float, dt: float,
                 steer_tau_s: float = 0.1, accel_tau_s: float = 0.2, max_steer_rad: float = 0.47):
        self.L, self.off, self.dt = wheelbase_m, rear_axle_offset_m, dt
        xr = start.x - rear_axle_offset_m * math.cos(start.yaw)
        yr = start.y - rear_axle_offset_m * math.sin(start.yaw)
        self.z = np.array([xr, yr, start.yaw, start.speed_mps])
        self.t = 0.0
        self.frame = 0
        self.a = 0.0
        self.delta = 0.0
        self.cmd = (0.0, 0.0)
        self.tau_d, self.tau_a, self.max_steer = steer_tau_s, accel_tau_s, max_steer_rad

    def state(self) -> EgoState:
        x, y, psi, v = self.z
        return EgoState(x + self.off * math.cos(psi), y + self.off * math.sin(psi), psi, v, self.t, self.frame)

    def apply(self, accel: float, delta: float, v_next: float | None = None) -> dict:
        self.cmd = (accel, float(np.clip(delta, -self.max_steer, self.max_steer)))
        return {"throttle": float("nan"), "steer": float("nan"), "brake": float("nan")}

    def step(self):
        self.a += (self.cmd[0] - self.a) * min(1.0, self.dt / self.tau_a)
        self.delta += (self.cmd[1] - self.delta) * min(1.0, self.dt / self.tau_d)
        self.z = bicycle_step(self.z, np.array([self.a, self.delta]), self.L, self.dt)
        self.t += self.dt
        self.frame += 1

    def collision_events(self) -> list:
        return []
