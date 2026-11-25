import math

import numpy as np
import pytest

from cone_navigation.actuation import ActuatorMapper
from cone_navigation.config import load_config
from cone_navigation.mpc import KinematicMPC, MPCConfig, bicycle_step, build_reference
from cone_navigation.planner import LocalPlanner, PlannerConfig
from cone_navigation.road import synthetic_straight_road
from cone_navigation.scenario import Region
from cone_navigation.state import EgoState, VehicleGeometry

CFG = load_config()
MCFG = MPCConfig.from_config(CFG)


def test_bicycle_straight_and_turn_sign():
    z = bicycle_step(np.array([0, 0, 0, 2.0]), np.array([0.0, 0.0]), 2.875, 0.1)
    assert z == pytest.approx([0.2, 0, 0, 2.0])
    z = bicycle_step(np.array([0, 0, 0, 2.0]), np.array([0.0, 0.1]), 2.875, 0.1)
    assert z[2] > 0                                   # positive wheel angle -> left (CCW) in map frame
    z = bicycle_step(np.array([0, 0, 0, 0.1]), np.array([-5.0, 0.0]), 2.875, 0.1)
    assert z[3] == 0.0                                # no reversing


def test_bicycle_circle_radius():
    L, delta, v, dt = 2.875, 0.2, 2.0, 0.001
    z = np.array([0, 0, 0, v])
    for _ in range(int(2 * math.pi * (L / math.tan(delta)) / v / dt)):
        z = bicycle_step(z, np.array([0, delta]), L, dt)
    assert math.hypot(z[0], z[1]) < 0.1               # back near the start after one circle


def _closed_loop(start: EgoState, seconds=8.0):
    road = synthetic_straight_road(85.0)
    geom = VehicleGeometry.from_config(CFG["vehicle"])
    plan = LocalPlanner(PlannerConfig.from_config(CFG), geom).plan(start, [], road,
                                                                   [Region("lane", 0, 85, -1.75, 1.75)])
    mpc = KinematicMPC(MCFG)
    off = MCFG.rear_axle_offset_m
    z = np.array([start.x - off * math.cos(start.yaw), start.y - off * math.sin(start.yaw), start.yaw, start.speed_mps])
    deltas, accels, ok = [], [], []
    for k in range(int(seconds / MCFG.dt)):
        ego = EgoState(z[0] + off * math.cos(z[2]), z[1] + off * math.sin(z[2]), z[2], z[3], k * MCFG.dt)
        res = mpc.solve(ego, build_reference(plan, ego, MCFG))
        ok.append(res.ok); deltas.append(res.delta); accels.append(res.accel)
        z = bicycle_step(z, np.array([res.accel, res.delta]), MCFG.wheelbase_m, MCFG.dt)
    return z, np.array(deltas), np.array(accels), ok


def test_mpc_converges_to_centreline_and_respects_bounds():
    # planned return to d=0 completes ~29 m ahead -> 12 s at <= 3 m/s
    z, deltas, accels, ok = _closed_loop(EgoState(0.0, 0.6, 0.05, 1.0, 0.0), seconds=12.0)
    assert all(ok)
    assert abs(z[1] + MCFG.rear_axle_offset_m * math.sin(z[2])) < 0.05   # origin back on d=0
    assert abs(z[3] - CFG["vehicle"]["target_speed_mps"]) < 0.1
    assert np.all(np.abs(deltas) <= MCFG.max_steer + 1e-6)
    assert np.all(np.abs(np.diff(deltas)) <= MCFG.max_steer_rate * MCFG.dt + 1e-6)
    assert np.all((accels <= MCFG.max_accel + 1e-6) & (accels >= -MCFG.max_brake - 1e-6))


def test_mpc_failure_falls_back_to_braking():
    mpc = KinematicMPC(MCFG)
    ego = EgoState(0, 0, 0, 3.0, 0)
    bad_ref = np.full((MCFG.horizon_steps + 1, 4), np.nan)
    res = mpc.solve(ego, bad_ref)
    assert not res.ok and res.fallback
    assert res.accel == pytest.approx(-MCFG.max_brake)


def test_actuator_mapping_signs_tables_and_rate_limit():
    m = ActuatorMapper(CFG["actuation"], CFG["vehicle"])
    # positive (left) wheel angle -> negative CARLA steer (steer_sign_map = -1)
    for _ in range(40):
        cmd = m.to_command(0.0, 0.1, 3.0)
    assert cmd.steer < 0
    assert m.delta_from_steer(cmd.steer) == pytest.approx(0.1, rel=1e-6)
    m.reset()
    cmd = m.to_command(0.0, 0.4, 3.0)
    assert abs(cmd.steer) == pytest.approx(CFG["actuation"]["max_steer_change_per_tick"])   # rate limited
    # calibrated points reproduce
    assert m.to_command(1.13, 0, 2.0).throttle == pytest.approx(0.4)
    brk = m.to_command(-0.18 - 2.27, 0, 2.0)
    assert brk.brake == pytest.approx(0.2) and brk.throttle == 0
    assert m.to_command(10.0, 0, 2.0).throttle <= CFG["actuation"]["max_throttle"]
    # standstill hold
    hold = m.to_command(0.0, 0.0, 0.1, v_next=0.0)
    assert hold.brake >= CFG["actuation"]["standstill_brake"] and hold.throttle == 0
