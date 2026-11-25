"""Road geometry container (pure data; built from CARLA or synthetically)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .frames import ReferenceLine


@dataclass(frozen=True)
class LaneInfo:
    name: str
    lane_id: int
    d_center: float         # road-frame lateral offset of the lane centre (left +)
    width: float
    same_direction: bool    # same travel direction as the ego lane

    @property
    def d_min(self) -> float:
        return self.d_center - self.width / 2

    @property
    def d_max(self) -> float:
        return self.d_center + self.width / 2


@dataclass
class RoadGeometry:
    reference: ReferenceLine            # ego-lane centreline, map frame
    lanes: list[LaneInfo]               # ego lane + neighbours, road frame
    source: str                         # "carla" or "synthetic"
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "meta": self.meta,
            "reference_xy_map": self.reference.xy.tolist(),
            "reference_z_carla": None if self.reference.z is None else self.reference.z.tolist(),
            "lanes": [vars(l) for l in self.lanes],
        }


def synthetic_straight_road(length_m: float = 85.0, spacing_m: float = 1.0,
                            lane_width_m: float = 3.5, origin=(0.0, 0.0),
                            heading_rad: float = 0.0) -> RoadGeometry:
    """Straight road with the same lane layout as the Town10HD test segment:
    ego lane (id -2), a same-direction lane to the left (id -1) and an
    opposite-direction lane beyond it (id 1). Used when CARLA is unavailable."""
    n = int(round(length_m / spacing_m)) + 1
    s = np.linspace(0.0, length_m, n)
    xy = np.stack([origin[0] + s * np.cos(heading_rad), origin[1] + s * np.sin(heading_rad)], axis=1)
    ref = ReferenceLine(xy, z=np.zeros(n))
    lanes = [
        LaneInfo("ego", -2, 0.0, lane_width_m, True),
        LaneInfo("left_1", -1, lane_width_m, lane_width_m, True),
        LaneInfo("left_2", 1, 2 * lane_width_m, lane_width_m, False),
    ]
    return RoadGeometry(ref, lanes, "synthetic", {"note": "synthetic straight road, no CARLA"})
