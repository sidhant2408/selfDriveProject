import math
from pathlib import Path

import numpy as np
import pytest
import yaml
from shapely.geometry import Polygon

from cone_navigation.config import load_config
from cone_navigation.planner import LocalPlanner, PlannerConfig, quintic_blend
from cone_navigation.road import synthetic_straight_road
from cone_navigation.scenario import load_scenario, resolve_cones, resolve_ego_start, scenario_from_dict
from cone_navigation.state import EgoState, VehicleGeometry, ego_footprint

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config()
GEOM = VehicleGeometry.from_config(CFG["vehicle"])
ROAD = synthetic_straight_road(85.0)


def plan_for(name, speed=0.0):
    scn = load_scenario(ROOT / "scenarios" / f"{name}.yaml")
    ego = resolve_ego_start(scn, ROAD)
    ego.speed_mps = speed
    planner = LocalPlanner(PlannerConfig.from_config(CFG), GEOM)
    cones = resolve_cones(scn, ROAD)
    return scn, cones, planner.plan(ego, cones, ROAD, scn.corridors)


def footprints_along(plan):
    for x, y, yaw in zip(plan.x, plan.y, plan.yaw):
        yield Polygon(ego_footprint(EgoState(x, y, yaw, 0, 0), GEOM))


def test_quintic_boundary_conditions():
    s = np.linspace(0, 30, 3001)
    d, d1, d2 = quintic_blend(s, 5.0, 20.0, 0.2, 3.0, slope_a=0.05)
    i0, i1 = np.searchsorted(s, 5.0), np.searchsorted(s, 25.0)
    assert d[i0] == pytest.approx(0.2, abs=1e-6) and d1[i0] == pytest.approx(0.05, abs=1e-3)
    assert d[i1] == pytest.approx(3.0, abs=1e-6) and d1[i1] == pytest.approx(0.0, abs=1e-3)
    assert d2[i1] == pytest.approx(0.0, abs=1e-3)
    # numerical derivative agrees with the analytic one
    np.testing.assert_allclose(np.gradient(d, s)[i0 + 10:i1 - 10], d1[i0 + 10:i1 - 10], atol=1e-3)


def test_empty_lane_keeps_centre():
    _, _, p = plan_for("empty_lane")
    assert p.decision == "PASS"
    assert np.max(np.abs(p.d)) < 1e-6


@pytest.mark.parametrize("name, margin", [("partial_obstruction", 0.40), ("tapered_diversion", 0.40)])
def test_pass_paths_clear_inflated_cones_with_full_body(name, margin):
    scn, cones, p = plan_for(name)
    assert p.decision == "PASS"
    inflated = [Polygon(c.footprint(margin)) for c in cones]
    for fp in footprints_along(p):
        assert not any(fp.intersects(c) for c in inflated)


def test_partial_obstruction_stays_in_ego_lane():
    scn, _, p = plan_for("partial_obstruction")
    for fp in footprints_along(p):
        s, d = ROAD.reference.to_frenet(*np.asarray(fp.exterior.coords).T)
        assert d.max() <= 1.75 + 1e-6 and d.min() >= -1.75 - 1e-6


def test_diversion_uses_permitted_left_lane():
    _, _, p = plan_for("tapered_diversion")
    assert p.d.max() > 3.0                         # moved into the left lane
    # never in the left lane before the corridor starts at s=10 (body corners)
    for fp, s in zip(footprints_along(p), p.s):
        cs, cd = ROAD.reference.to_frenet(*np.asarray(fp.exterior.coords).T)
        assert np.all((cd <= 1.75 + 1e-6) | (cs >= 10.0 - 1e-6))


def test_closure_stops_before_barrier_with_feasible_profile():
    scn, cones, p = plan_for("full_closure", speed=3.0)
    assert p.decision == "STOP"
    barrier_s = min(ROAD.reference.to_frenet(*c.footprint(0.4).T)[0].min() for c in cones)
    assert p.s_stop + GEOM.length_m / 2 < barrier_s           # front bumper short of the inflated barrier
    assert np.all(np.diff(p.v[p.s <= p.s_stop]) <= 1e-9)       # monotone slow-down
    assert np.all(p.v[p.s >= p.s_stop] == 0)
    assert p.required_decel_mps2 <= CFG["planner"]["max_decel_mps2"]


def test_forbidden_corridor_forces_stop():
    """A path exists physically (left lane) but is not permitted -> must STOP."""
    raw = yaml.safe_load((ROOT / "scenarios" / "tapered_diversion.yaml").read_text())
    raw["corridors"] = [raw["corridors"][0]]                   # ego lane only
    scn = scenario_from_dict(raw)
    planner = LocalPlanner(PlannerConfig.from_config(CFG), GEOM)
    p = planner.plan(resolve_ego_start(scn, ROAD), resolve_cones(scn, ROAD), ROAD, scn.corridors)
    assert p.decision == "STOP"


def test_replanning_commits_to_previous_path():
    scn = load_scenario(ROOT / "scenarios" / "tapered_diversion.yaml")
    cones = resolve_cones(scn, ROAD)
    planner = LocalPlanner(PlannerConfig.from_config(CFG), GEOM)
    first = planner.plan(resolve_ego_start(scn, ROAD), cones, ROAD, scn.corridors)
    x, y, yaw = ROAD.reference.to_world(1.0, float(np.interp(1.0, first.s, first.d)))
    second = planner.plan(EgoState(x, y, yaw, 3.0, 0.3), cones, ROAD, scn.corridors)
    assert "continue previous" in second.reason
    np.testing.assert_allclose(np.interp(second.s, first.s, first.d), second.d, atol=1e-6)
