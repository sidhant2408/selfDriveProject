import math

import numpy as np
import pytest
from shapely.geometry import Polygon

from cone_navigation.observations import ConeObservation
from cone_navigation.state import EgoState, VehicleGeometry, ego_footprint, oriented_rect

GEOM = VehicleGeometry(4.694, 2.088, 2.875, -0.0685, math.radians(35))


def test_oriented_rect_axis_aligned_ccw():
    c = oriented_rect(0, 0, 0, 4, 2)
    np.testing.assert_allclose(c, [[2, 1], [-2, 1], [-2, -1], [2, -1]])
    assert Polygon(c).exterior.is_ccw
    assert Polygon(c).area == pytest.approx(8.0)


def test_oriented_rect_rotated():
    c = oriented_rect(1, 1, math.pi / 2, 4, 2)   # long axis along +y
    xs, ys = c[:, 0], c[:, 1]
    assert (xs.max() - xs.min(), ys.max() - ys.min()) == pytest.approx((2.0, 4.0))


def test_ego_footprint_dimensions_and_offset():
    fp = ego_footprint(EgoState(10, 5, 0.0, 0, 0), GEOM)
    assert fp[:, 0].max() - fp[:, 0].min() == pytest.approx(GEOM.length_m)
    assert fp[:, 1].max() - fp[:, 1].min() == pytest.approx(GEOM.width_m)
    assert fp[:, 0].mean() == pytest.approx(10 + GEOM.bbox_center_offset_x_m)
    big = ego_footprint(EgoState(10, 5, 0.0, 0, 0), GEOM, buffer_m=0.3)
    assert Polygon(big).area == pytest.approx((GEOM.length_m + 0.6) * (GEOM.width_m + 0.6))


def test_fallen_cone_footprint_is_long_along_yaw():
    up = ConeObservation("u", 0, 0, 0.0, "upright", 0.48, 0.48)
    fallen = ConeObservation("f", 0, 0, math.pi / 2, "fallen", 0.72, 0.48)
    fp = fallen.footprint()
    assert fp[:, 1].max() - fp[:, 1].min() == pytest.approx(0.72)   # along +y (yaw 90 deg)
    assert fp[:, 0].max() - fp[:, 0].min() == pytest.approx(0.48)
    assert Polygon(fallen.footprint()).area > Polygon(up.footprint()).area
    assert Polygon(up.footprint(0.4)).area == pytest.approx(1.28 ** 2)


def test_cone_center_distance_is_not_clearance():
    # A cone centre 1.2 m from the ego centre "looks" outside the car if you
    # only compare against half-width (1.044 m), but its 0.24 m radius overlaps.
    ego = Polygon(ego_footprint(EgoState(0, 0, 0, 0, 0), GEOM))
    cone = Polygon(ConeObservation("c", 0, 1.2, 0, "upright", 0.48, 0.48).footprint())
    assert ego.intersects(cone)


def test_invalid_cone_rejected():
    with pytest.raises(ValueError):
        ConeObservation("x", 0, 0, 0, "sideways", 0.5, 0.5)
    with pytest.raises(ValueError):
        ConeObservation("x", 0, 0, 0, "upright", 0.0, 0.5)
