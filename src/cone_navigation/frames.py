"""Coordinate frames and conversions.

Four frames exist in this project. Never mix them.

1. CARLA world (``*_c``): Unreal's LEFT-handed frame. x forward/east, y to the
   RIGHT, z up, metres. Rotation.yaw is in DEGREES and increases CLOCKWISE seen
   from above. Only ``carla_*.py`` adapters see this frame.

2. Map frame (``M``): the right-handed planar frame every algorithm uses.
       x_m = x_c,   y_m = -y_c,   yaw_m = -radians(yaw_c)   (wrapped to [-pi, pi))
   yaw_m is counter-clockwise from +x_m, radians. This is the same mirror CARLA's
   own ``Transform.to_right_handed()`` applies (FLU, REP-103). The kinematic
   bicycle model x += v cos(psi) dt, y += v sin(psi) dt is valid here as written.

3. Road frame ``(s, d)``: relative to a ReferenceLine (a polyline in M).
   s = arc length along the line from its first point, d = signed lateral
   offset, POSITIVE TO THE LEFT of the direction of travel.

4. Ego body frame: x forward, y left (right-handed), origin at the CARLA actor
   origin. The bounding-box centre is offset by ``bbox_center_offset_x_m``.
"""
from __future__ import annotations

import math

import numpy as np


def wrap_angle(a):
    """Wrap angle(s) in radians to [-pi, pi). Returns float for scalar input."""
    r = (np.asarray(a, dtype=float) + math.pi) % (2.0 * math.pi) - math.pi
    return float(r) if np.ndim(a) == 0 else r


def carla_to_map_xy(x_c: float, y_c: float) -> tuple[float, float]:
    return x_c, -y_c


def map_to_carla_xy(x_m: float, y_m: float) -> tuple[float, float]:
    return x_m, -y_m


def carla_yaw_deg_to_map_rad(yaw_deg: float) -> float:
    return float(wrap_angle(-math.radians(yaw_deg)))


def map_yaw_rad_to_carla_deg(yaw_rad: float) -> float:
    return -math.degrees(yaw_rad)


class ReferenceLine:
    """Piecewise-linear reference polyline in the map frame.

    ``to_frenet`` projects onto the nearest segment; the first and last
    segments are extended linearly so points slightly before/after the line
    still get a well-defined (s, d) (s < 0 or s > length).
    """

    def __init__(self, xy, z=None):
        xy = np.asarray(xy, dtype=float)
        if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2:
            raise ValueError("ReferenceLine needs an (N>=2, 2) array of map-frame points")
        seg = np.diff(xy, axis=0)
        seg_len = np.hypot(seg[:, 0], seg[:, 1])
        keep = np.concatenate([[True], seg_len > 1e-6])  # drop duplicate points
        if not keep.all():
            xy = xy[keep]
            z = None if z is None else np.asarray(z, dtype=float)[keep]
            seg = np.diff(xy, axis=0)
            seg_len = np.hypot(seg[:, 0], seg[:, 1])
        if len(xy) < 2:
            raise ValueError("ReferenceLine has fewer than 2 distinct points")
        self.xy = xy
        self.z = None if z is None else np.asarray(z, dtype=float)
        self.seg_len = seg_len
        self.unit = seg / seg_len[:, None]
        self.heading = np.arctan2(self.unit[:, 1], self.unit[:, 0])
        self.s = np.concatenate([[0.0], np.cumsum(seg_len)])

    @property
    def length(self) -> float:
        return float(self.s[-1])

    def _segment_index(self, s):
        idx = np.searchsorted(self.s, s, side="right") - 1
        return np.clip(idx, 0, len(self.seg_len) - 1)

    def heading_at(self, s):
        return self.heading[self._segment_index(np.asarray(s, dtype=float))]

    def z_at(self, s):
        """Road-surface CARLA z interpolated along s (0.0 if unknown)."""
        if self.z is None:
            return np.zeros_like(np.asarray(s, dtype=float))
        return np.interp(s, self.s, self.z)

    def to_world(self, s, d):
        """(s, d) -> map-frame (x, y, yaw). Accepts scalars or arrays."""
        scalar = np.ndim(s) == 0 and np.ndim(d) == 0
        s = np.atleast_1d(np.asarray(s, dtype=float))
        d = np.atleast_1d(np.asarray(d, dtype=float))
        s, d = np.broadcast_arrays(s, d)
        i = self._segment_index(s)
        t = s - self.s[i]
        u = self.unit[i]
        p = self.xy[i] + t[:, None] * u
        left = np.stack([-u[:, 1], u[:, 0]], axis=1)
        out = p + d[:, None] * left
        yaw = self.heading[i]
        if scalar:
            return float(out[0, 0]), float(out[0, 1]), float(yaw[0])
        return out[:, 0], out[:, 1], yaw

    def to_frenet(self, x, y):
        """Map-frame (x, y) -> (s, d). Accepts scalars or arrays."""
        scalar = np.ndim(x) == 0 and np.ndim(y) == 0
        p = np.stack(np.broadcast_arrays(np.atleast_1d(np.asarray(x, float)),
                                         np.atleast_1d(np.asarray(y, float))), axis=1)
        a = self.xy[:-1]                                    # (K, 2)
        rel = p[:, None, :] - a[None, :, :]                 # (M, K, 2)
        t = np.einsum("mkj,kj->mk", rel, self.unit)         # along-segment
        lo = np.zeros(len(self.seg_len)); lo[0] = -np.inf
        hi = self.seg_len.copy(); hi[-1] = np.inf
        tc = np.clip(t, lo, hi)
        proj = a[None] + tc[..., None] * self.unit[None]
        dist = np.linalg.norm(p[:, None, :] - proj, axis=2)
        k = np.argmin(dist, axis=1)
        m = np.arange(len(p))
        s = self.s[k] + tc[m, k]
        u = self.unit[k]
        r = rel[m, k]
        d = u[:, 0] * r[:, 1] - u[:, 1] * r[:, 0]           # cross(u, r): left +
        if scalar:
            return float(s[0]), float(d[0])
        return s, d
