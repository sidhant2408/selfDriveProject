"""Deterministic synthetic cone arrangements for CARLA-free development/tests.

Each function returns a list of (cone_id, s, d, state) tuples in the road frame
of ``road.synthetic_straight_road`` (ego lane d in [-1.75, 1.75], same-direction
left lane d in [1.75, 5.25]). These are the fixtures the Phase 2 classifier will
be tested against. They are SYNTHETIC algorithm inputs, not a CARLA scene.
"""
from __future__ import annotations

import numpy as np

from .observations import ConeObservation
from .road import RoadGeometry

ConeTuple = tuple[str, float, float, str]


def partial_obstruction() -> list[ConeTuple]:
    """A: cones along the right edge of the ego lane; ego lane stays passable."""
    return [(f"a{i}", 25.0 + 2.0 * i, -1.35, "upright") for i in range(5)]


def tapered_diversion() -> list[ConeTuple]:
    """B: diagonal taper closing the ego lane, then a straight run along the
    lane line; traffic is guided into the left (same-direction) lane."""
    taper = [(f"b{i}", 20.0 + 3.0 * i, -1.5 + 0.6 * i, "upright") for i in range(6)]  # d: -1.5 -> 1.5
    run = [(f"b{6 + i}", 38.0 + 4.0 * i, 1.6, "upright") for i in range(4)]
    return taper + run


def full_closure() -> list[ConeTuple]:
    """C: a line of cones across every lane at s = 40 (incl. one fallen)."""
    ds = np.arange(-1.5, 5.25, 0.75)
    cones = [(f"c{i}", 40.0, float(d), "upright") for i, d in enumerate(ds)]
    cones[3] = (cones[3][0], 40.0, cones[3][2], "fallen")
    return cones


def scattered(seed: int = 0, n: int = 6) -> list[ConeTuple]:
    """Ambiguous: random scatter (no line, no taper). Seeded => repeatable."""
    rng = np.random.default_rng(seed)
    s = np.sort(rng.uniform(20.0, 45.0, n))
    d = rng.uniform(-1.5, 4.5, n)
    return [(f"x{i}", float(si), float(di), "upright") for i, (si, di) in enumerate(zip(s, d))]


def to_observations(cones: list[ConeTuple], road: RoadGeometry,
                    upright_dims=(0.48, 0.48), fallen_dims=(0.72, 0.48)) -> list[ConeObservation]:
    out = []
    for cid, s, d, state in cones:
        x, y, yaw = road.reference.to_world(s, d)
        length, width = fallen_dims if state == "fallen" else upright_dims
        out.append(ConeObservation(cid, x, y, yaw, state, length, width))
    return out
