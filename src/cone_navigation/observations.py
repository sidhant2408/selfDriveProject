"""Common cone observation schema.

Every cone source (ground truth now; noisy adapter in Phase 5; camera/LiDAR in
Phase 6) must produce exactly this type. Downstream modules (pattern detector,
planner) consume only ``ConeObservation`` lists, never CARLA actors.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .state import oriented_rect

ConeState = Literal["upright", "fallen", "unknown"]
VALID_STATES = ("upright", "fallen", "unknown")


@dataclass
class ConeObservation:
    """One cone, MAP frame.

    x, y   : footprint centre (for a fallen cone: middle of the lying cone,
             not its base).
    yaw    : rad, direction of the footprint's long axis. For a fallen cone it
             points from base to tip. Irrelevant for an upright cone.
    length : footprint extent along yaw (fallen: cone height), metres.
    width  : footprint extent across yaw (base diameter), metres.
    covariance_xy : 2x2 position covariance in the map frame (zeros for GT).
    """
    cone_id: str
    x: float
    y: float
    yaw: float
    state: ConeState
    length: float
    width: float
    covariance_xy: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))

    def __post_init__(self):
        if self.state not in VALID_STATES:
            raise ValueError(f"cone {self.cone_id}: invalid state {self.state!r}")
        if self.length <= 0 or self.width <= 0:
            raise ValueError(f"cone {self.cone_id}: footprint dimensions must be > 0")
        self.covariance_xy = np.asarray(self.covariance_xy, dtype=float).reshape(2, 2)

    def footprint(self, margin_m: float = 0.0) -> np.ndarray:
        """Footprint rectangle corners (4, 2), inflated by margin on all sides."""
        return oriented_rect(self.x, self.y, self.yaw,
                             self.length + 2 * margin_m, self.width + 2 * margin_m)

    def to_dict(self) -> dict:
        return {
            "cone_id": self.cone_id, "x": self.x, "y": self.y, "yaw": self.yaw,
            "state": self.state, "length": self.length, "width": self.width,
            "covariance_xy": self.covariance_xy.tolist(),
        }
