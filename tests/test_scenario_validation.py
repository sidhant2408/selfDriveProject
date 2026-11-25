import copy
import math
from pathlib import Path

import pytest
import yaml

from cone_navigation.road import synthetic_straight_road
from cone_navigation.scenario import (ScenarioError, load_scenario, resolve_cones,
                                      scenario_from_dict, validate_geometry)
from cone_navigation.state import VehicleGeometry

ROOT = Path(__file__).resolve().parents[1]
SCN_A = ROOT / "scenarios" / "partial_obstruction.yaml"
GEOM = VehicleGeometry(4.694, 2.088, 2.875, -0.0685, math.radians(35))


@pytest.fixture
def raw():
    return yaml.safe_load(SCN_A.read_text())


@pytest.mark.parametrize("fname, n_cones, label", [
    ("partial_obstruction.yaml", 5, "PARTIAL"),
    ("tapered_diversion.yaml", 10, "DIVERSION"),
    ("full_closure.yaml", 9, "CLOSURE"),
])
def test_scenario_files_load_and_validate(fname, n_cones, label):
    scn = load_scenario(ROOT / "scenarios" / fname)
    assert scn.map_name == "Town10HD_Opt" and len(scn.cones) == n_cones
    assert scn.ground_truth_label == label
    road = synthetic_straight_road(scn.route.length_m)
    assert validate_geometry(scn, road, GEOM) == []


def test_closure_fallen_cone_lies_across_road():
    scn = load_scenario(ROOT / "scenarios" / "full_closure.yaml")
    road = synthetic_straight_road(scn.route.length_m)
    fallen = [c for c in resolve_cones(scn, road) if c.state == "fallen"]
    assert len(fallen) == 1
    fp = fallen[0].footprint()
    _, d = road.reference.to_frenet(fp[:, 0], fp[:, 1])
    assert d.max() - d.min() == pytest.approx(0.72)   # long axis across the road


def test_resolution_is_deterministic(raw):
    road = synthetic_straight_road(85.0)
    a = resolve_cones(scenario_from_dict(raw), road)
    b = resolve_cones(scenario_from_dict(copy.deepcopy(raw)), road)
    assert [(c.x, c.y, c.yaw) for c in a] == [(c.x, c.y, c.yaw) for c in b]


@pytest.mark.parametrize("mutate, needle", [
    (lambda d: d["cones"].append(dict(d["cones"][0])), "duplicate cone id"),
    (lambda d: d["cones"][0].update(state="sideways"), "state must be"),
    (lambda d: d["corridors"][0].update(d_min=2.0, d_max=1.0), "d range is empty or inverted"),
    (lambda d: d.pop("map"), "missing required key 'map'"),
    (lambda d: d.update(corridors=[]), "at least one permitted corridor"),
    (lambda d: d["cones"][0].update(s="far"), "expected a finite number"),
    (lambda d: d.update(ground_truth_label="EASY"), "ground_truth_label"),
])
def test_load_time_errors(raw, mutate, needle):
    mutate(raw)
    with pytest.raises(ScenarioError) as e:
        scenario_from_dict(raw)
    assert any(needle in p for p in e.value.problems), e.value.problems


@pytest.mark.parametrize("mutate, needle", [
    (lambda d: d["cones"][0].update(s=200.0), "outside route"),
    (lambda d: d["cones"][0].update(s=1.0, d=0.0), "overlaps the ego start"),
    (lambda d: d["cones"][1].update(s=25.1), "overlap"),
    (lambda d: d["ego"].update(d=1.5), "not inside any permitted corridor"),
    (lambda d: d["corridors"][0].update(d_max=20.0), "leaves the mapped lanes"),
    (lambda d: d["goal"].update(s_max=120.0), "goal"),
])
def test_geometry_errors(raw, mutate, needle):
    mutate(raw)
    scn = scenario_from_dict(raw)
    problems = validate_geometry(scn, synthetic_straight_road(85.0), GEOM)
    assert any(needle in p for p in problems), problems


def test_all_problems_reported_at_once(raw):
    raw["cones"][0]["state"] = "bad"
    raw.pop("map")
    with pytest.raises(ScenarioError) as e:
        scenario_from_dict(raw)
    assert len(e.value.problems) >= 2
