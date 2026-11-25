"""Bird's-eye plots (matplotlib, headless Agg backend)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch, Polygon as MplPolygon  # noqa: E402

from .observations import ConeObservation  # noqa: E402
from .road import RoadGeometry  # noqa: E402
from .state import EgoState, VehicleGeometry, ego_footprint  # noqa: E402

# Role colours (reference palette, light mode).
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
EGO, CONE, FALLEN, CORRIDOR, GOAL = "#2a78d6", "#eb6834", "#e34948", "#1baf7a", "#6250d6"
LANE_FILL, OPPOSITE_FILL = "#f0efec", "#e2e1dc"


def _region_polygon(road: RoadGeometry, s0, s1, d0, d1, n=40) -> np.ndarray:
    """Road-frame box -> map-frame polygon (follows the reference line)."""
    s = np.linspace(s0, s1, n)
    xl, yl, _ = road.reference.to_world(s, np.full(n, d1))
    xr, yr, _ = road.reference.to_world(s[::-1], np.full(n, d0))
    return np.column_stack([np.concatenate([xl, xr]), np.concatenate([yl, yr])])


def plot_scene(road: RoadGeometry, cones: list[ConeObservation], ego: EgoState,
               geom: VehicleGeometry, corridors, goal, title: str, out_path: str | Path,
               margin_m: float = 0.0, synthetic: bool = False) -> Path:
    """Two panels: map frame (equal aspect) and road frame (s, d)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    L = road.reference.length
    fig, (ax_m, ax_r) = plt.subplots(2, 1, figsize=(12, 8.5), facecolor=SURFACE,
                                     gridspec_kw={"height_ratios": [1, 1]})

    # Every drawable element as (s/d-region, colour) is drawn in both frames.
    def draw(ax, frame: str):
        ax.set_facecolor(SURFACE)
        to_xy = (lambda s, d: road.reference.to_world(s, d)[:2]) if frame == "map" else (lambda s, d: (s, d))

        def region(s0, s1, d0, d1, **kw):
            if frame == "map":
                ax.add_patch(MplPolygon(_region_polygon(road, s0, s1, d0, d1), closed=True, **kw))
            else:
                ax.add_patch(MplPolygon([[s0, d0], [s1, d0], [s1, d1], [s0, d1]], closed=True, **kw))

        for lane in road.lanes:
            region(0, L, lane.d_min, lane.d_max,
                   facecolor=LANE_FILL if lane.same_direction else OPPOSITE_FILL,
                   edgecolor=INK2, linewidth=0.8, hatch=None if lane.same_direction else "//")
        for cor in corridors:
            region(cor.s_min, cor.s_max, cor.d_min, cor.d_max, facecolor=CORRIDOR, alpha=0.18,
                   edgecolor=CORRIDOR, linewidth=1.5)
        region(goal.s_min, goal.s_max, goal.d_min, goal.d_max, facecolor="none",
               edgecolor=GOAL, linewidth=2.0, linestyle="--")
        sx, sy = to_xy(np.linspace(0, L, 200), np.zeros(200))
        ax.plot(sx, sy, color=INK2, linewidth=1.0, linestyle=(0, (6, 4)))

        def poly(corners, **kw):
            pts = np.asarray(corners)
            if frame == "road":
                s, d = road.reference.to_frenet(pts[:, 0], pts[:, 1])
                pts = np.column_stack([s, d])
            ax.add_patch(MplPolygon(pts, closed=True, **kw))

        for c in cones:
            col = FALLEN if c.state == "fallen" else CONE
            if margin_m > 0:
                poly(c.footprint(margin_m), facecolor="none", edgecolor=col, linewidth=1.0, linestyle=":")
            poly(c.footprint(), facecolor=col, edgecolor=SURFACE, linewidth=1.0)
        poly(ego_footprint(ego, geom), facecolor=EGO, alpha=0.85, edgecolor=SURFACE, linewidth=1.0)
        arrow_len = geom.length_m * 0.6
        if frame == "map":
            hx, hy, rel = ego.x, ego.y, ego.yaw
        else:
            hx, hy = road.reference.to_frenet(ego.x, ego.y)
            rel = float(ego.yaw - road.reference.heading_at(hx))
        ax.arrow(hx, hy, arrow_len * np.cos(rel), arrow_len * np.sin(rel),
                 width=0.12, color=INK, length_includes_head=True)

        ax.grid(True, color="#d8d7d2", linewidth=0.5)
        ax.tick_params(colors=INK2, labelsize=9)
        for sp in ax.spines.values():
            sp.set_color("#bdbcb6")

    draw(ax_m, "map")
    ax_m.set_aspect("equal")
    ax_m.autoscale_view()
    ax_m.set_xlabel("x_map [m]  (= x_carla)", color=INK2)
    ax_m.set_ylabel("y_map [m]  (= -y_carla)", color=INK2)
    ax_m.set_title("Map frame (right-handed, bird's-eye)", color=INK, fontsize=10, loc="left")

    draw(ax_r, "road")
    dmin = min(l.d_min for l in road.lanes) - 0.5
    dmax = max(l.d_max for l in road.lanes) + 0.5
    ax_r.set_xlim(-5, L + 2)
    ax_r.set_ylim(dmin, dmax)
    ax_r.set_xlabel("s [m] along ego-lane centre", color=INK2)
    ax_r.set_ylabel("d [m] (left +)", color=INK2)
    ax_r.set_title("Road frame (s, d) - axes not to scale", color=INK, fontsize=10, loc="left")

    handles = [
        Patch(facecolor=EGO, label="ego footprint (bbox)"),
        Patch(facecolor=CONE, label="cone footprint (upright)"),
        Patch(facecolor=FALLEN, label="cone footprint (fallen)"),
        Patch(facecolor=CORRIDOR, alpha=0.35, edgecolor=CORRIDOR, label="permitted corridor"),
        Patch(facecolor="none", edgecolor=GOAL, linestyle="--", label="goal region"),
        Patch(facecolor=LANE_FILL, edgecolor=INK2, label="lane, same direction"),
        Patch(facecolor=OPPOSITE_FILL, edgecolor=INK2, hatch="//", label="lane, opposite direction"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9)
    tag = "SYNTHETIC ROAD - no CARLA" if synthetic else "CARLA scene (spawned actors)"
    fig.suptitle(f"{title}   [{tag}]", color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.savefig(out_path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def plot_episode(road: RoadGeometry, truth: list[ConeObservation], rows: list[dict], plans: list,
                 geom: VehicleGeometry, corridors, goal, margin_m: float, title: str,
                 out_path: str | Path) -> Path:
    """Road-frame overview (true cones, footprints every 1 s, planned paths) +
    speed tracking + commands."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax, av, au) = plt.subplots(3, 1, figsize=(12, 10.5), facecolor=SURFACE,
                                     gridspec_kw={"height_ratios": [1.3, 1, 1]})
    L = road.reference.length
    for lane in road.lanes:
        ax.add_patch(MplPolygon([[0, lane.d_min], [L, lane.d_min], [L, lane.d_max], [0, lane.d_max]],
                                facecolor=LANE_FILL if lane.same_direction else OPPOSITE_FILL,
                                edgecolor=INK2, linewidth=0.8, hatch=None if lane.same_direction else "//"))
    for c in corridors:
        ax.add_patch(MplPolygon([[c.s_min, c.d_min], [c.s_max, c.d_min], [c.s_max, c.d_max], [c.s_min, c.d_max]],
                                facecolor=CORRIDOR, alpha=0.15, edgecolor=CORRIDOR, linewidth=1.2))
    ax.add_patch(MplPolygon([[goal.s_min, goal.d_min], [goal.s_max, goal.d_min], [goal.s_max, goal.d_max],
                             [goal.s_min, goal.d_max]], facecolor="none", edgecolor=GOAL, linewidth=2, linestyle="--"))

    def to_sd(pts):
        s, d = road.reference.to_frenet(pts[:, 0], pts[:, 1])
        return np.column_stack([s, d])

    for c in truth:
        col = FALLEN if c.state == "fallen" else CONE
        ax.add_patch(MplPolygon(to_sd(c.footprint(margin_m)), facecolor="none", edgecolor=col, linestyle=":", linewidth=1))
        ax.add_patch(MplPolygon(to_sd(c.footprint()), facecolor=col, edgecolor=SURFACE, linewidth=0.8))
    for i, (_, p) in enumerate(plans):
        ax.plot(p.s, p.d, color=INK2, linewidth=0.8, linestyle=(0, (3, 3)), alpha=0.6,
                label="planned path (snapshots)" if i == 0 else None)
        if p.s_stop is not None:
            ax.axvline(p.s_stop, color=FALLEN, linewidth=1.2, linestyle="--",
                       label="planned stop (ego centre)" if i == len(plans) - 1 else None)
    s = np.array([r["s"] for r in rows]); d = np.array([r["d"] for r in rows])
    t = np.array([r["t_s"] for r in rows])
    step = max(1, int(round(1.0 / max(t[1] - t[0], 1e-3)))) if len(t) > 1 else 1
    for r in rows[::step] + rows[-1:]:
        e = EgoState(r["x_map"], r["y_map"], r["yaw_map"], 0, 0)
        ax.add_patch(MplPolygon(to_sd(ego_footprint(e, geom)), facecolor="none", edgecolor=EGO, linewidth=0.9, alpha=0.7))
    ax.plot(s, d, color=EGO, linewidth=2, label="ego origin trajectory")
    ax.set_xlim(-5, L + 2)
    ax.set_ylim(min(l.d_min for l in road.lanes) - 0.5, max(l.d_max for l in road.lanes) + 0.5)
    ax.set_xlabel("s [m]", color=INK2); ax.set_ylabel("d [m] (left +)", color=INK2)
    ax.set_title("Road frame: true cones (solid), margin (dotted), ego bbox every 1 s - axes not to scale",
                 loc="left", fontsize=10, color=INK)
    ax.legend(loc="upper right", frameon=False, fontsize=8)

    av.plot(t, [r["speed_mps"] for r in rows], color=EGO, linewidth=2, label="speed (measured)")
    av.plot(t, [r["v_ref"] for r in rows], color=INK2, linewidth=1.5, linestyle="--", label="v_ref (MPC horizon start)")
    av.set_ylabel("m/s", color=INK2); av.legend(frameon=False, fontsize=8, loc="upper right")
    av.set_title("Speed tracking", loc="left", fontsize=10, color=INK)

    au.plot(t, [r["accel_cmd"] for r in rows], color=CORRIDOR, linewidth=1.8, label="MPC accel [m/s^2]")
    au.plot(t, [r["delta_cmd"] * 10 for r in rows], color=CONE, linewidth=1.8, label="MPC wheel angle x10 [rad]")
    au.plot(t, [r["cross_track_m"] for r in rows], color=GOAL, linewidth=1.5, label="cross-track error [m]")
    au.axhline(0, color=INK2, linewidth=0.6)
    au.set_xlabel("sim time [s]", color=INK2); au.legend(frameon=False, fontsize=8, loc="upper right")
    au.set_title("MPC commands and tracking error", loc="left", fontsize=10, color=INK)
    for a in (ax, av, au):
        a.set_facecolor(SURFACE); a.grid(True, color="#d8d7d2", linewidth=0.5)
        a.tick_params(colors=INK2, labelsize=9)
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def plot_smoke_log(rows: list[dict], out_path: str | Path, title: str) -> Path:
    """Trajectory + speed + controls/yaw-rate from the smoke-test log."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t = np.array([r["t_s"] for r in rows])
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), facecolor=SURFACE)
    ax = axes[0]
    ax.plot([r["x_map"] for r in rows], [r["y_map"] for r in rows], color=EGO, linewidth=2)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x_map [m]", color=INK2); ax.set_ylabel("y_map [m]", color=INK2)
    ax.set_title("Ego path (map frame)", loc="left", fontsize=10, color=INK)
    ax = axes[1]
    ax.plot(t, [r["speed_mps"] for r in rows], color=EGO, linewidth=2, label="speed [m/s]")
    ax.plot(t, [r["throttle"] for r in rows], color=CORRIDOR, linewidth=2, label="throttle cmd")
    ax.plot(t, [r["brake"] for r in rows], color=FALLEN, linewidth=2, label="brake cmd")
    ax.set_xlabel("sim time [s]", color=INK2); ax.legend(frameon=False, fontsize=9)
    ax.set_title("Speed and longitudinal commands", loc="left", fontsize=10, color=INK)
    ax = axes[2]
    ax.plot(t, [r["steer"] for r in rows], color=CONE, linewidth=2, label="steer cmd (normalised)")
    ax.plot(t, [r["yaw_rate_map_rps"] for r in rows], color=GOAL, linewidth=2, label="yaw rate, map frame [rad/s]")
    ax.axhline(0, color=INK2, linewidth=0.6)
    ax.set_xlabel("sim time [s]", color=INK2); ax.legend(frameon=False, fontsize=9)
    ax.set_title("Steering sign check", loc="left", fontsize=10, color=INK)
    for a in axes:
        a.set_facecolor(SURFACE); a.grid(True, color="#d8d7d2", linewidth=0.5)
        a.tick_params(colors=INK2, labelsize=9)
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, facecolor=SURFACE)
    plt.close(fig)
    return out_path
