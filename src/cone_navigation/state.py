"""Vehicle state and footprint geometry (map frame, metres, radians)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VehicleGeometry:
    length_m: float
    width_m: float
    wheelbase_m: float
    bbox_center_offset_x_m: float
    max_wheel_steer_rad: float

    @classmethod
    def from_config(cls, vehicle_cfg: dict) -> "VehicleGeometry":
        return cls(
            length_m=float(vehicle_cfg["length_m"]),
            width_m=float(vehicle_cfg["width_m"]),
            wheelbase_m=float(vehicle_cfg["wheelbase_m"]),
            bbox_center_offset_x_m=float(vehicle_cfg.get("bbox_center_offset_x_m", 0.0)),
            max_wheel_steer_rad=math.radians(float(vehicle_cfg["max_wheel_steer_deg"])),
        )


@dataclass
class EgoState:
    """Ego state in the MAP frame. x, y = CARLA actor origin (converted)."""
    x: float
    y: float
    yaw: float              # rad, CCW from +x_map
    speed_mps: float        # signed, along body x
    timestamp_s: float      # simulation time
    frame: int | None = None


def oriented_rect(cx: float, cy: float, yaw: float, length: float, width: float) -> np.ndarray:
    """Corners (4, 2) of a rectangle centred at (cx, cy), long axis along yaw.

    Order is counter-clockwise: front-left, rear-left, rear-right, front-right.
    """
    c, s = math.cos(yaw), math.sin(yaw)
    hl, hw = length / 2.0, width / 2.0
    local = np.array([[hl, hw], [-hl, hw], [-hl, -hw], [hl, -hw]])
    rot = np.array([[c, -s], [s, c]])
    return local @ rot.T + np.array([cx, cy])


def ego_footprint(state: EgoState, geom: VehicleGeometry, buffer_m: float = 0.0) -> np.ndarray:
    """Conservative ego rectangle (bounding box + optional buffer on all sides)."""
    cx = state.x + geom.bbox_center_offset_x_m * math.cos(state.yaw)
    cy = state.y + geom.bbox_center_offset_x_m * math.sin(state.yaw)
    return oriented_rect(cx, cy, state.yaw, geom.length_m + 2 * buffer_m, geom.width_m + 2 * buffer_m)
