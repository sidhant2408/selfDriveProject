"""CARLA-free checks of the synthetic fixtures Phase 2 will classify.

These verify the *fixtures* are geometrically what they claim to be (computed
directly here, independent of any classifier). They are not classifier tests.
"""
import math
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

from cone_navigation import synthetic
from cone_navigation.ground_truth import GroundTruthConeProvider
from cone_navigation.road import synthetic_straight_road
from cone_navigation.scenario import load_scenario
from cone_navigation.state import EgoState, VehicleGeometry
from cone_navigation.visualization import plot_scene

ROAD = synthetic_straight_road(85.0)
GEOM = VehicleGeometry(4.694, 2.088, 2.875, -0.0685, math.radians(35))
EGO_LANE = (-1.75, 1.75)
MARGIN = 0.40


def lateral_extent(obs):
    fp = obs.footprint(MARGIN)
    _, d = ROAD.reference.to_frenet(fp[:, 0], fp[:, 1])
    return d.min(), d.max()


def test_partial_obstruction_leaves_passable_gap():
    obs = synthetic.to_observations(synthetic.partial_obstruction(), ROAD)
    blocked_up_to = max(lateral_extent(o)[1] for o in obs)       # cones hug the right edge
    free = EGO_LANE[1] - blocked_up_to
    assert free > GEOM.width_m, f"free width {free:.2f} m"


def test_full_closure_blocks_every_lane():
    obs = synthetic.to_observations(synthetic.full_closure(), ROAD)
    # Every free lateral gap (between inflated cones and at both drivable edges)
    # must be narrower than the car.
    covered = sorted(lateral_extent(o) for o in obs)
    road_lo = min(l.d_min for l in ROAD.lanes if l.same_direction)
    road_hi = max(l.d_max for l in ROAD.lanes if l.same_direction)
    gaps, hi = [], road_lo
    for a, b in covered:
        gaps.append(max(0.0, a - hi))
        hi = max(hi, b)
    gaps.append(max(0.0, road_hi - hi))
    assert max(gaps) < GEOM.width_m, gaps
    assert any(o.state == "fallen" for o in obs)


def test_taper_is_monotonic_and_scatter_is_not():
    taper = synthetic.tapered_diversion()[:6]
    d = np.array([c[2] for c in taper])
    assert np.all(np.diff(d) > 0)
    scat = synthetic.scattered(seed=3)
    assert scat == synthetic.scattered(seed=3)                    # seeded => repeatable
    ds = np.diff([c[2] for c in scat])
    assert not (np.all(ds > 0) or np.all(ds < 0))


def test_provider_returns_copies():
    obs = synthetic.to_observations(synthetic.partial_obstruction(), ROAD)
    p = GroundTruthConeProvider(obs)
    got = p.get_cones()
    got[0].x += 100.0
    assert p.get_cones()[0].x == obs[0].x


def test_offline_plot_renders(tmp_path):
    scn = load_scenario(Path(__file__).resolve().parents[1] / "scenarios" / "partial_obstruction.yaml")
    obs = synthetic.to_observations(synthetic.partial_obstruction(), ROAD)
    out = plot_scene(ROAD, obs, EgoState(0, 0, 0, 0, 0), GEOM, scn.corridors, scn.goal,
                     "test", tmp_path / "scene.png", margin_m=MARGIN, synthetic=True)
    assert out.is_file() and out.stat().st_size > 10_000
    # Sanity: no cone footprint intersects the ego at the start.
    assert not any(Polygon(o.footprint()).intersects(Polygon([[-2.4, -1.05], [2.4, -1.05],
                                                              [2.4, 1.05], [-2.4, 1.05]])) for o in obs)
