"""Scenario loading, validation and (s, d) -> map-frame resolution.

Pure module: no CARLA import. CARLA spawning lives in ``carla_world.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from shapely.geometry import Polygon

from .observations import VALID_STATES, ConeObservation
from .road import RoadGeometry
from .state import EgoState, VehicleGeometry, ego_footprint


class ScenarioError(ValueError):
    """Raised with every problem found, not just the first."""

    def __init__(self, problems: list[str], source: str = "<dict>"):
        self.problems = problems
        super().__init__(f"Invalid scenario {source}:\n  - " + "\n  - ".join(problems))


@dataclass(frozen=True)
class RouteSpec:
    anchor_x_carla: float
    anchor_y_carla: float
    expected_road_id: int | None
    expected_lane_id: int | None
    length_m: float
    sample_spacing_m: float


@dataclass(frozen=True)
class EgoSpec:
    s: float
    d: float
    spawn_z_offset_m: float


@dataclass(frozen=True)
class Region:
    """Axis-aligned box in the road frame."""
    name: str
    s_min: float
    s_max: float
    d_min: float
    d_max: float

    def contains(self, s: float, d: float) -> bool:
        return self.s_min <= s <= self.s_max and self.d_min <= d <= self.d_max


@dataclass(frozen=True)
class ConeSpec:
    cone_id: str
    s: float
    d: float
    state: str
    yaw_offset_rad: float   # relative to road heading at s (fallen-cone direction)
    length: float
    width: float


@dataclass
class Scenario:
    name: str
    description: str
    map_name: str
    seed: int
    route: RouteSpec
    ego: EgoSpec
    goal: Region
    corridors: list[Region]
    cone_blueprint: str
    cones: list[ConeSpec]
    ground_truth_label: str | None = None   # evaluation only; never read by pipeline
    source: str = "<dict>"
    raw: dict = field(default_factory=dict, repr=False)


# --------------------------------------------------------------------------- load

def load_scenario(path: str | Path) -> Scenario:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Scenario file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return scenario_from_dict(data, source=str(path))


def scenario_from_dict(data: dict, source: str = "<dict>") -> Scenario:
    problems: list[str] = []
    if not isinstance(data, dict):
        raise ScenarioError(["top level must be a mapping"], source)

    def req(d: dict, key: str, ctx: str):
        if not isinstance(d, dict) or key not in d:
            problems.append(f"{ctx}: missing required key '{key}'")
            return None
        return d[key]

    def num(v, ctx: str, positive: bool = False):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            problems.append(f"{ctx}: expected a finite number, got {v!r}")
            return float("nan")
        if positive and v <= 0:
            problems.append(f"{ctx}: must be > 0, got {v}")
        return float(v)

    name = req(data, "name", "scenario") or "unnamed"
    map_name = req(data, "map", "scenario")
    if map_name is not None and not isinstance(map_name, str):
        problems.append("map: must be a string")
    seed = data.get("seed", 0)
    if not isinstance(seed, int) or isinstance(seed, bool):
        problems.append(f"seed: expected int, got {seed!r}")

    r = req(data, "route", "scenario") or {}
    anchor = req(r, "anchor", "route") or {}
    route = RouteSpec(
        anchor_x_carla=num(req(anchor, "x", "route.anchor"), "route.anchor.x"),
        anchor_y_carla=num(req(anchor, "y", "route.anchor"), "route.anchor.y"),
        expected_road_id=r.get("expected_road_id"),
        expected_lane_id=r.get("expected_lane_id"),
        length_m=num(req(r, "length_m", "route"), "route.length_m", positive=True),
        sample_spacing_m=num(r.get("sample_spacing_m", 1.0), "route.sample_spacing_m", positive=True),
    )

    e = req(data, "ego", "scenario") or {}
    ego = EgoSpec(
        s=num(e.get("s", 0.0), "ego.s"),
        d=num(e.get("d", 0.0), "ego.d"),
        spawn_z_offset_m=num(e.get("spawn_z_offset_m", 0.3), "ego.spawn_z_offset_m"),
    )

    def region(d: dict, ctx: str, name_: str) -> Region:
        if "s_start" in d or "s_end" in d:
            s_min, s_max = d.get("s_start"), d.get("s_end")
        else:
            s_min, s_max = d.get("s_min"), d.get("s_max")
        reg = Region(name_, num(s_min, f"{ctx}.s_min/s_start"), num(s_max, f"{ctx}.s_max/s_end"),
                     num(d.get("d_min"), f"{ctx}.d_min"), num(d.get("d_max"), f"{ctx}.d_max"))
        if not reg.s_min < reg.s_max:
            problems.append(f"{ctx}: s range is empty or inverted ({reg.s_min}..{reg.s_max})")
        if not reg.d_min < reg.d_max:
            problems.append(f"{ctx}: d range is empty or inverted ({reg.d_min}..{reg.d_max})")
        return reg

    goal = region(req(data, "goal", "scenario") or {}, "goal", "goal")

    corridors = []
    cor_list = data.get("corridors") or []
    if not cor_list:
        problems.append("corridors: at least one permitted corridor is required")
    for i, c in enumerate(cor_list):
        corridors.append(region(c, f"corridors[{i}]", str(c.get("name", f"corridor_{i}"))))

    defaults = data.get("cone_defaults") or {}
    blueprint = defaults.get("blueprint", "static.prop.constructioncone")
    dims = {
        "upright": defaults.get("upright", {"length": 0.48, "width": 0.48}),
        "fallen": defaults.get("fallen", {"length": 0.72, "width": 0.48}),
    }
    dims["unknown"] = dims["upright"]

    cones: list[ConeSpec] = []
    seen: set[str] = set()
    for i, c in enumerate(data.get("cones") or []):
        ctx = f"cones[{i}]"
        cid = str(c.get("id", f"cone_{i}"))
        if cid in seen:
            problems.append(f"{ctx}: duplicate cone id '{cid}'")
        seen.add(cid)
        state = c.get("state", "upright")
        if state not in VALID_STATES or state == "unknown":
            problems.append(f"{ctx}: state must be 'upright' or 'fallen', got {state!r}")
            state = "upright"
        dim = dims[state]
        cones.append(ConeSpec(
            cone_id=cid,
            s=num(c.get("s"), f"{ctx}.s"),
            d=num(c.get("d"), f"{ctx}.d"),
            state=state,
            yaw_offset_rad=math.radians(num(c.get("yaw_offset_deg", 0.0), f"{ctx}.yaw_offset_deg")),
            length=num(c.get("length", dim.get("length")), f"{ctx}.length", positive=True),
            width=num(c.get("width", dim.get("width")), f"{ctx}.width", positive=True),
        ))

    label = data.get("ground_truth_label")
    if label is not None and label not in ("PARTIAL", "DIVERSION", "CLOSURE", "UNKNOWN"):
        problems.append(f"ground_truth_label: unexpected value {label!r}")

    if problems:
        raise ScenarioError(problems, source)
    return Scenario(name=name, description=str(data.get("description", "")).strip(),
                    map_name=map_name, seed=seed, route=route, ego=ego, goal=goal,
                    corridors=corridors, cone_blueprint=blueprint, cones=cones,
                    ground_truth_label=label, source=source, raw=data)


# ---------------------------------------------------------------- resolve / check

def resolve_cones(scn: Scenario, road: RoadGeometry) -> list[ConeObservation]:
    """Ground-truth cone placements in the map frame (deterministic)."""
    out = []
    for c in scn.cones:
        x, y, heading = road.reference.to_world(c.s, c.d)
        out.append(ConeObservation(c.cone_id, x, y, heading + c.yaw_offset_rad,
                                   c.state, c.length, c.width))
    return out


def resolve_ego_start(scn: Scenario, road: RoadGeometry) -> EgoState:
    x, y, yaw = road.reference.to_world(scn.ego.s, scn.ego.d)
    return EgoState(x, y, yaw, 0.0, 0.0)


def validate_geometry(scn: Scenario, road: RoadGeometry, geom: VehicleGeometry) -> list[str]:
    """Checks needing the road and vehicle. Returns problems (empty = OK)."""
    problems: list[str] = []
    L = road.reference.length
    for c in scn.cones:
        if not 0.0 <= c.s <= L:
            problems.append(f"cone {c.cone_id}: s={c.s} outside route [0, {L:.1f}]")
    if scn.goal.s_min < 0.0 or scn.goal.s_max > L + 1e-6:
        problems.append(f"goal: s range {scn.goal.s_min}..{scn.goal.s_max} outside route [0, {L:.1f}]")
    if not 0.0 <= scn.ego.s < L:
        problems.append(f"ego: s={scn.ego.s} outside route")

    # Ego must start fully inside a permitted corridor.
    half_w = geom.width_m / 2
    if not any(cor.contains(scn.ego.s, scn.ego.d - half_w) and cor.contains(scn.ego.s, scn.ego.d + half_w)
               for cor in scn.corridors):
        problems.append("ego: start footprint is not inside any permitted corridor")

    # Ego must not start overlapping a cone.
    ego_poly = Polygon(ego_footprint(resolve_ego_start(scn, road), geom))
    cones = resolve_cones(scn, road)
    for obs in cones:
        if ego_poly.intersects(Polygon(obs.footprint())):
            problems.append(f"cone {obs.cone_id}: overlaps the ego start footprint")

    # Cones overlapping each other are almost certainly a typo.
    for i in range(len(cones)):
        for j in range(i + 1, len(cones)):
            if Polygon(cones[i].footprint()).intersects(Polygon(cones[j].footprint())):
                problems.append(f"cones {cones[i].cone_id} and {cones[j].cone_id} overlap")

    # Corridors must lie on the mapped road.
    lanes_dmin = min(l.d_min for l in road.lanes)
    lanes_dmax = max(l.d_max for l in road.lanes)
    for cor in scn.corridors:
        if cor.d_min < lanes_dmin - 1e-6 or cor.d_max > lanes_dmax + 1e-6:
            problems.append(f"corridor {cor.name}: d range leaves the mapped lanes "
                            f"[{lanes_dmin:.2f}, {lanes_dmax:.2f}]")
    return problems
