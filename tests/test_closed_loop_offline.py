"""End-to-end planner + MPC on the kinematic plant (no CARLA). Slowest tests."""
from pathlib import Path

import pytest

from cone_navigation.config import load_config
from cone_navigation.ground_truth import GroundTruthConeProvider
from cone_navigation.mpc import KinematicMPC, MPCConfig
from cone_navigation.planner import LocalPlanner, PlannerConfig
from cone_navigation.plant import KinematicPlant
from cone_navigation.road import synthetic_straight_road
from cone_navigation.runner import expected_outcome, run_episode
from cone_navigation.scenario import load_scenario, resolve_cones, resolve_ego_start
from cone_navigation.state import VehicleGeometry

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config()


@pytest.mark.parametrize("name", ["empty_lane", "partial_obstruction", "tapered_diversion", "full_closure"])
def test_episode_outcome(name):
    scn = load_scenario(ROOT / "scenarios" / f"{name}.yaml")
    road = synthetic_straight_road(scn.route.length_m)
    geom = VehicleGeometry.from_config(CFG["vehicle"])
    mcfg = MPCConfig.from_config(CFG)
    truth = resolve_cones(scn, road)
    plant = KinematicPlant(resolve_ego_start(scn, road), mcfg.wheelbase_m, mcfg.rear_axle_offset_m, 0.05)
    r = run_episode(plant, GroundTruthConeProvider(truth), LocalPlanner(PlannerConfig.from_config(CFG), geom),
                    KinematicMPC(mcfg), road, scn, geom, truth, 0.05, 40.0, 4)
    s = r.summary
    assert r.outcome == expected_outcome(scn.ground_truth_label)
    assert s["mpc_failures"] == 0
    assert s["max_corridor_violation_m"] == 0.0
    if truth:
        assert s["min_true_clearance_m"] > 0.3          # margin 0.40 minus tracking slack
    assert s["steer_rate_cmd_max_rps"] <= mcfg.max_steer_rate + 1e-6
