import math

import numpy as np
import pytest

from cone_navigation.frames import (ReferenceLine, carla_to_map_xy, carla_yaw_deg_to_map_rad,
                                    map_to_carla_xy, map_yaw_rad_to_carla_deg, wrap_angle)


def test_carla_map_xy_roundtrip_and_mirror():
    assert carla_to_map_xy(3.0, 4.0) == (3.0, -4.0)
    assert map_to_carla_xy(*carla_to_map_xy(-24.3, -57.8)) == (-24.3, -57.8)


def test_yaw_conventions():
    # CARLA yaw +90 deg points along +y_carla, which is -y_map => map yaw -pi/2.
    assert carla_yaw_deg_to_map_rad(90.0) == pytest.approx(-math.pi / 2)
    assert carla_yaw_deg_to_map_rad(0.0) == pytest.approx(0.0)
    # 180 and -180 are the same heading; wrapped to [-pi, pi).
    assert carla_yaw_deg_to_map_rad(180.0) == pytest.approx(-math.pi)
    for deg in (-170.0, -30.0, 0.6, 45.0, 179.0):
        back = map_yaw_rad_to_carla_deg(carla_yaw_deg_to_map_rad(deg))
        assert wrap_angle(math.radians(back - deg)) == pytest.approx(0.0, abs=1e-12)


def test_heading_vector_consistent_with_position_mirror():
    # A unit step along CARLA heading, mirrored, must equal the step along map heading.
    yaw_c = 33.0
    dx_c, dy_c = math.cos(math.radians(yaw_c)), math.sin(math.radians(yaw_c))
    yaw_m = carla_yaw_deg_to_map_rad(yaw_c)
    assert (dx_c, -dy_c) == pytest.approx((math.cos(yaw_m), math.sin(yaw_m)))


def test_wrap_angle_scalar_and_array():
    assert wrap_angle(3 * math.pi) == pytest.approx(-math.pi)
    np.testing.assert_allclose(wrap_angle(np.array([0.0, 2 * math.pi + 0.1])), [0.0, 0.1], atol=1e-12)


def test_straight_line_frenet_left_positive():
    ref = ReferenceLine([[0, 0], [10, 0], [20, 0]])
    s, d = ref.to_frenet(12.0, 2.0)
    assert (s, d) == pytest.approx((12.0, 2.0))       # +y_map is LEFT of +x travel
    s, d = ref.to_frenet(5.0, -1.5)
    assert (s, d) == pytest.approx((5.0, -1.5))
    # Extrapolation beyond both ends.
    assert ref.to_frenet(-3.0, 1.0) == pytest.approx((-3.0, 1.0))
    assert ref.to_frenet(25.0, 0.0) == pytest.approx((25.0, 0.0))


def test_curved_line_roundtrip():
    theta = np.linspace(0, math.pi / 2, 91)
    R = 40.0
    ref = ReferenceLine(np.column_stack([R * np.sin(theta), R * (1 - np.cos(theta))]))  # left turn
    rng = np.random.default_rng(0)
    s = rng.uniform(1.0, ref.length - 1.0, 200)
    d = rng.uniform(-3.0, 3.0, 200)
    x, y, _ = ref.to_world(s, d)
    s2, d2 = ref.to_frenet(x, y)
    np.testing.assert_allclose(s2, s, atol=0.05)   # polyline corners -> tiny ambiguity
    # Inside a polyline bend, points near a vertex can project onto the
    # neighbouring segment: sub-mm d error, inherent to piecewise-linear lines.
    np.testing.assert_allclose(d2, d, atol=1e-3)
    # On a left turn, d>0 is toward the centre of curvature (0, R).
    x0, y0, _ = ref.to_world(ref.length / 2, 2.0)
    assert math.hypot(x0, y0 - R) == pytest.approx(R - 2.0, abs=0.05)


def test_duplicate_points_and_bad_input():
    ref = ReferenceLine([[0, 0], [0, 0], [1, 0]])
    assert ref.length == pytest.approx(1.0)
    with pytest.raises(ValueError):
        ReferenceLine([[0, 0]])
