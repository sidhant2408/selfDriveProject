"""MPC (acceleration, wheel angle) -> normalised CARLA throttle / brake / steer.

Pure module: returns plain numbers; the CARLA plant builds VehicleControl.
Calibrated tables live in configs/default.yaml (`actuation`), measured with
scripts/calibrate_actuators.py. They are low-speed fits for one vehicle; the
simulator remains the source of the true dynamics.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class ActuatorCommand:
    throttle: float
    steer: float          # normalised [-1, 1], CARLA sign convention
    brake: float
    a_effective: float    # acceleration actually mapped (after speed feedback)


class ActuatorMapper:
    def __init__(self, act_cfg: dict, vehicle_cfg: dict):
        t, b = act_cfg["throttle_table"], act_cfg["brake_table"]
        self.thr_tab = np.asarray(t["throttle"], float)
        self.thr_acc = np.asarray(t["accel_mps2"], float)
        self.brk_tab = np.asarray(b["brake"], float)
        self.brk_dec = np.asarray(b["decel_mps2"], float)
        if np.any(np.diff(self.thr_acc) <= 0) or np.any(np.diff(self.brk_dec) <= 0):
            raise ValueError("actuation tables must be strictly increasing")
        self.coast = float(act_cfg.get("coast_accel_mps2", -0.18))
        self.max_throttle = float(act_cfg.get("max_throttle", 0.6))
        self.steer_gain = float(act_cfg.get("steer_gain", 1.0))
        self.max_dsteer = float(act_cfg.get("max_steer_change_per_tick", 0.05))
        self.k_v = float(act_cfg.get("speed_feedback_gain", 0.0))
        self.standstill_brake = float(act_cfg.get("standstill_brake", 0.4))
        self.sign = int(vehicle_cfg.get("steer_sign_map", -1))
        self.max_wheel = math.radians(float(vehicle_cfg["max_wheel_steer_deg"]))
        self.last_steer = 0.0

    def reset(self):
        self.last_steer = 0.0

    def steer_from_delta(self, delta: float) -> float:
        """Map-frame wheel angle (left +) -> CARLA normalised steer (unsaturated)."""
        return self.sign * delta / (self.steer_gain * self.max_wheel)

    def delta_from_steer(self, steer: float) -> float:
        return steer * self.steer_gain * self.max_wheel / self.sign

    def to_command(self, accel: float, delta: float, v_now: float, v_next: float | None = None) -> ActuatorCommand:
        # steering: convert, saturate, rate-limit per physics tick
        s = float(np.clip(self.steer_from_delta(delta), -1.0, 1.0))
        s = float(np.clip(s, self.last_steer - self.max_dsteer, self.last_steer + self.max_dsteer))
        self.last_steer = s

        a = accel
        if v_next is not None:
            a += self.k_v * (v_next - v_now)          # trim steady-state speed error
        if a >= self.coast:
            thr = float(np.interp(a, self.thr_acc, self.thr_tab))
            cmd = ActuatorCommand(min(thr, self.max_throttle), s, 0.0, a)
        else:
            brk = float(np.interp(self.coast - a, self.brk_dec, self.brk_tab, right=1.0))
            cmd = ActuatorCommand(0.0, s, min(brk, 1.0), a)
        if v_next is not None and v_next < 0.05 and v_now < 0.3:
            cmd.throttle, cmd.brake = 0.0, max(cmd.brake, self.standstill_brake)   # hold at standstill
        return cmd
