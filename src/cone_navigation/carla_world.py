"""CARLA adapters: map -> RoadGeometry, scenario spawning, ego state readout.

All conversions between the CARLA (left-handed) frame and the map frame happen
here via ``frames.py``; nothing downstream sees a carla object.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import carla
import numpy as np

from .carla_session import CarlaSession, CarlaSetupError
from .frames import (ReferenceLine, carla_to_map_xy, carla_yaw_deg_to_map_rad,
                     map_to_carla_xy, map_yaw_rad_to_carla_deg)
from .observations import ConeObservation
from .road import LaneInfo, RoadGeometry
from .scenario import RouteSpec
from .state import EgoState


def _yaw_diff_deg(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def build_road_geometry(cmap: carla.Map, route: RouteSpec, max_heading_change_deg: float = 2.0) -> RoadGeometry:
    """Sample the ego-lane centreline from the anchor along the travel direction."""
    anchor = carla.Location(x=route.anchor_x_carla, y=route.anchor_y_carla, z=0.0)
    wp = cmap.get_waypoint(anchor, project_to_road=True, lane_type=carla.LaneType.Driving)
    if wp is None:
        raise CarlaSetupError(f"No driving lane near anchor {anchor}")
    snap = wp.transform.location.distance(carla.Location(anchor.x, anchor.y, wp.transform.location.z))
    if snap > 2.0:
        raise CarlaSetupError(f"Anchor is {snap:.1f} m from the nearest driving lane centre")
    if route.expected_road_id is not None and wp.road_id != route.expected_road_id:
        raise CarlaSetupError(f"Anchor lane is road {wp.road_id}, scenario expects {route.expected_road_id}")
    if route.expected_lane_id is not None and wp.lane_id != route.expected_lane_id:
        raise CarlaSetupError(f"Anchor lane is lane {wp.lane_id}, scenario expects {route.expected_lane_id}")

    pts, zs, widths = [], [], []
    cur, travelled = wp, 0.0
    yaw0 = wp.transform.rotation.yaw
    while True:
        loc = cur.transform.location
        pts.append(carla_to_map_xy(loc.x, loc.y)); zs.append(loc.z); widths.append(cur.lane_width)
        if travelled >= route.length_m - 1e-6:
            break
        nxt = cur.next(route.sample_spacing_m)
        if not nxt:
            raise CarlaSetupError(f"Lane ends after {travelled:.1f} m (< {route.length_m} m)")
        nxt = min(nxt, key=lambda w: _yaw_diff_deg(w.transform.rotation.yaw, cur.transform.rotation.yaw))
        if nxt.is_junction:
            raise CarlaSetupError(f"Junction reached after {travelled:.1f} m; shorten route.length_m")
        if _yaw_diff_deg(nxt.transform.rotation.yaw, yaw0) > max_heading_change_deg:
            raise CarlaSetupError(f"Road bends > {max_heading_change_deg} deg after {travelled:.1f} m; "
                                  "the MVP expects a straight segment")
        cur, travelled = nxt, travelled + route.sample_spacing_m

    ref = ReferenceLine(np.array(pts), z=np.array(zs))
    if max(widths) - min(widths) > 0.1:
        print(f"[road] WARNING: ego lane width varies {min(widths):.2f}..{max(widths):.2f} m")

    # Neighbour lanes, measured at mid-route so projection is well inside the line.
    mid = cmap.get_waypoint(carla.Location(*map_to_carla_xy(*ref.to_world(ref.length / 2, 0.0)[:2]), 0.0),
                            lane_type=carla.LaneType.Driving)
    # Direction is decided by HEADING, not lane-id sign: ids flip sign between
    # OpenDRIVE roads (this route crosses road 7 -> another road id mid-way).
    mid_yaw = mid.transform.rotation.yaw
    lanes = [LaneInfo("ego", wp.lane_id, 0.0, float(np.median(widths)), True)]
    for side in ("left", "right"):
        w, k = mid, 0
        while True:
            # get_left/right_lane is relative to each lane's OWN direction; for a
            # lane running opposite to us, its "right" is our left.
            w_same = _yaw_diff_deg(w.transform.rotation.yaw, mid_yaw) < 90.0
            nb = (w.get_left_lane() if w_same else w.get_right_lane()) if side == "left" \
                else (w.get_right_lane() if w_same else w.get_left_lane())
            if nb is None or nb.lane_type != carla.LaneType.Driving:
                break
            same = _yaw_diff_deg(nb.transform.rotation.yaw, mid_yaw) < 90.0
            k += 1
            s_n, d_n = ref.to_frenet(*carla_to_map_xy(nb.transform.location.x, nb.transform.location.y))
            lanes.append(LaneInfo(f"{side}_{k}", nb.lane_id, round(float(d_n), 3), nb.lane_width, same))
            w = nb
            if k > 6:
                break
    lanes.sort(key=lambda l: l.d_center)
    meta = {"map": cmap.name, "road_id": wp.road_id, "lane_id": wp.lane_id,
            "anchor_snapped_carla": [wp.transform.location.x, wp.transform.location.y,
                                     wp.transform.location.z],
            "anchor_yaw_carla_deg": wp.transform.rotation.yaw}
    return RoadGeometry(ref, lanes, "carla", meta)


# ----------------------------------------------------------------------- ego

def spawn_ego(sess: CarlaSession, blueprint_id: str, start: EgoState, z_road: float, z_offset: float):
    bp = sess.blueprint(blueprint_id)
    bp.set_attribute("role_name", "hero")
    x_c, y_c = map_to_carla_xy(start.x, start.y)
    tf = carla.Transform(carla.Location(x=x_c, y=y_c, z=z_road + z_offset),
                         carla.Rotation(yaw=map_yaw_rad_to_carla_deg(start.yaw)))
    ego = sess.spawn(bp, tf)
    if ego is None:
        raise CarlaSetupError(f"Could not spawn ego '{blueprint_id}' at {tf} (location occupied?)")
    ego.set_autopilot(False)  # our controller is the only command source
    return ego


def read_ego_state(sess: CarlaSession, ego) -> EgoState:
    """Frame-consistent state from the world snapshot of the latest tick."""
    snap = sess.world.get_snapshot()
    a = snap.find(ego.id)
    tf, vel = a.get_transform(), a.get_velocity()
    x, y = carla_to_map_xy(tf.location.x, tf.location.y)
    fwd = tf.get_forward_vector()
    speed = vel.x * fwd.x + vel.y * fwd.y + vel.z * fwd.z  # signed, along body x
    return EgoState(x, y, carla_yaw_deg_to_map_rad(tf.rotation.yaw), float(speed),
                    float(snap.timestamp.elapsed_seconds), int(snap.frame))


# --------------------------------------------------------------------- cones

@dataclass
class ConeRecord:
    """Registry entry kept at spawn time (props may not be retrievable later)."""
    cone_id: str
    blueprint: str
    state: str
    visual: bool                    # False => synthetic record, no actor in the scene
    actor_id: int | None
    requested_carla: dict           # x, y, z, pitch, yaw, roll
    actual_carla: dict | None       # read back from the actor after settling
    actor_bbox_extent: list | None
    truth_map: dict                 # ConeObservation.to_dict() in the map frame

    def to_dict(self) -> dict:
        return asdict(self)


def _tf_dict(tf: carla.Transform) -> dict:
    return {"x": tf.location.x, "y": tf.location.y, "z": tf.location.z,
            "pitch": tf.rotation.pitch, "yaw": tf.rotation.yaw, "roll": tf.rotation.roll}


def cone_carla_transform(obs: ConeObservation, z_road: float) -> carla.Transform:
    """Prop transform for a cone observation.

    Upright: actor origin = base centre on the road surface (measured: the prop
    origin is at the base). Fallen: Rotation(pitch=-90, yaw=Y) lays the cone's
    up-axis horizontally along CARLA yaw Y (checked with Rotation.get_up_vector);
    base = footprint centre - length/2 along yaw, raised by the base radius.
    The fallen visual pose has NOT been checked against a camera image yet.
    """
    yaw_c = map_yaw_rad_to_carla_deg(obs.yaw)
    if obs.state == "fallen":
        bx = obs.x - 0.5 * obs.length * math.cos(obs.yaw)
        by = obs.y - 0.5 * obs.length * math.sin(obs.yaw)
        x_c, y_c = map_to_carla_xy(bx, by)
        return carla.Transform(carla.Location(x=x_c, y=y_c, z=z_road + obs.width / 2),
                               carla.Rotation(pitch=-90.0, yaw=yaw_c))
    x_c, y_c = map_to_carla_xy(obs.x, obs.y)
    return carla.Transform(carla.Location(x=x_c, y=y_c, z=z_road), carla.Rotation(yaw=yaw_c))


def spawn_cones(sess: CarlaSession, truths: list[ConeObservation], blueprint_id: str,
                road: RoadGeometry, allow_synthetic: bool = False) -> list[ConeRecord]:
    try:
        bp = sess.blueprint(blueprint_id)
    except CarlaSetupError:
        if not allow_synthetic:
            raise
        print(f"[cones] WARNING: blueprint {blueprint_id} missing -> SYNTHETIC cone records, "
              "nothing visible in the scene")
        bp = None
    records = []
    for obs in truths:
        s, _ = road.reference.to_frenet(obs.x, obs.y)
        tf = cone_carla_transform(obs, float(road.reference.z_at(s)))
        actor = sess.spawn(bp, tf) if bp is not None else None
        if actor is None and bp is not None:
            msg = f"cone {obs.cone_id}: spawn failed at {tf}"
            if not allow_synthetic:
                raise CarlaSetupError(msg + " (use --allow-synthetic-cones to continue without visuals)")
            print(f"[cones] WARNING: {msg} -> synthetic record")
        records.append(ConeRecord(obs.cone_id, blueprint_id, obs.state, actor is not None,
                                  actor.id if actor else None, _tf_dict(tf), None,
                                  None, obs.to_dict()))
    return records


def read_back_cones(sess: CarlaSession, records: list[ConeRecord]) -> float:
    """Fill actual transforms/bboxes; return max horizontal drift from request [m]."""
    drift = 0.0
    for r in records:
        if not r.visual:
            continue
        a = sess.world.get_actor(r.actor_id)
        tf = a.get_transform()
        r.actual_carla = _tf_dict(tf)
        e = a.bounding_box.extent
        r.actor_bbox_extent = [e.x, e.y, e.z]
        drift = max(drift, math.hypot(tf.location.x - r.requested_carla["x"],
                                      tf.location.y - r.requested_carla["y"]))
    return drift

