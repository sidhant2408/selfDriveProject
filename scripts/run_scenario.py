"""Phase 1: load a scenario, place cones, record exact poses, plot a bird's-eye view.

CARLA mode (default): spawns ego + cone props, settles, reads every pose back,
destroys everything, repeats (--repeats, default 2) and checks that the
re-run reproduces identical cone locations.

Offline mode (--offline): no simulator; uses a synthetic straight road with the
same lane layout. Output is clearly tagged SYNTHETIC.

Usage:
    python scripts/run_scenario.py scenarios/partial_obstruction.yaml
    python scripts/run_scenario.py scenarios/partial_obstruction.yaml --offline
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

from cone_navigation.config import PROJECT_ROOT, load_config
from cone_navigation.ground_truth import GroundTruthConeProvider
from cone_navigation.road import synthetic_straight_road
from cone_navigation.scenario import (ScenarioError, load_scenario, resolve_cones,
                                      resolve_ego_start, validate_geometry)
from cone_navigation.state import VehicleGeometry
from cone_navigation.visualization import plot_scene


def run_carla(scn, cfg, geom, repeats: int, settle_ticks: int, allow_synthetic: bool,
              hold_s: float = 0.0, view: str = "topdown"):
    from cone_navigation.carla_session import CarlaSession
    from cone_navigation.carla_world import (build_road_geometry, read_back_cones,
                                             read_ego_state, spawn_cones, spawn_ego)
    import carla

    runs = []
    with CarlaSession(cfg["simulation"], map_name=scn.map_name, realtime=hold_s > 0) as sess:
        print(f"[scenario] CARLA client {sess.versions['client']} / server {sess.versions['server']}, "
              f"map {sess.map.name}")
        road = build_road_geometry(sess.map, scn.route)
        lanes_txt = ", ".join(f"{l.name}(id {l.lane_id}, d={l.d_center:+.2f}, "
                              f"{'same' if l.same_direction else 'opposite'})" for l in road.lanes)
        print(f"[scenario] reference line {road.reference.length:.1f} m on road {road.meta['road_id']}; "
              f"lanes: {lanes_txt}")
        problems = validate_geometry(scn, road, geom)
        if problems:
            raise ScenarioError(problems, scn.source)
        mid_x, mid_y = road.reference.to_world(road.reference.length / 2, 1.75)[:2]
        sess.set_spectator_topdown(mid_x, -mid_y, 60.0)

        for r in range(repeats):
            truths = resolve_cones(scn, road)
            start = resolve_ego_start(scn, road)
            ego = spawn_ego(sess, cfg["vehicle"]["blueprint"], start,
                            float(road.reference.z_at(scn.ego.s)), scn.ego.spawn_z_offset_m)
            records = spawn_cones(sess, truths, scn.cone_blueprint, road, allow_synthetic)
            if view == "chase":                    # behind/above the ego, looking at the cones
                sess.follow_spectator(ego, back_m=7.0, up_m=6.0, pitch_deg=-22.0)
            # Hold ego braked while it settles; with --hold-seconds (first run only)
            # keep ticking in real time so the window stays live for viewing.
            n_hold = settle_ticks + (int(hold_s / sess.fixed_dt) if r == 0 else 0)
            if r == 0 and hold_s > 0:
                print(f"[scenario] holding scene for {hold_s:.0f} s - look at the CARLA window")
            for _ in range(n_hold):
                ego.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                sess.tick()
            drift = read_back_cones(sess, records)
            ego_state = read_ego_state(sess, ego)
            n_vis = sum(rec.visual for rec in records)
            print(f"[scenario] run {r}: {n_vis}/{len(records)} cones spawned as visible props, "
                  f"max cone drift after {settle_ticks} ticks = {drift * 100:.2f} cm, "
                  f"ego at s,d = {tuple(round(v, 3) for v in road.reference.to_frenet(ego_state.x, ego_state.y))}")
            runs.append({"records": records, "ego": ego_state, "drift_m": drift, "truths": truths})
            sess.destroy_actors()
    return road, runs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenario", nargs="?", default=str(PROJECT_ROOT / "scenarios" / "partial_obstruction.yaml"))
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    ap.add_argument("--offline", action="store_true", help="no CARLA: synthetic straight road")
    ap.add_argument("--repeats", type=int, default=2, help="spawn/reset cycles to check repeatability")
    ap.add_argument("--settle-ticks", type=int, default=20)
    ap.add_argument("--allow-synthetic-cones", action="store_true",
                    help="continue with invisible cone records if the prop cannot be spawned")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "outputs"))
    ap.add_argument("--hold-seconds", type=float, default=0.0,
                    help="keep the spawned scene up (real-time ticking) so you can look at it")
    ap.add_argument("--view", choices=["chase", "topdown"], default="topdown", help="spectator camera")
    args = ap.parse_args()

    cfg = load_config(args.config)
    scn = load_scenario(args.scenario)
    random.seed(scn.seed); np.random.seed(scn.seed)
    geom = VehicleGeometry.from_config(cfg["vehicle"])
    margin = float(cfg.get("uncertainty", {}).get("fixed_margin_m", 0.0))
    out_dir = Path(args.out) / f"{scn.name}_{'offline' if args.offline else 'carla'}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[scenario] {scn.name}: {len(scn.cones)} cones, map {scn.map_name}, seed {scn.seed}")

    if args.offline:
        road = synthetic_straight_road(scn.route.length_m, scn.route.sample_spacing_m)
        problems = validate_geometry(scn, road, geom)
        if problems:
            raise ScenarioError(problems, scn.source)
        runs = [{"records": None, "ego": resolve_ego_start(scn, road), "drift_m": 0.0,
                 "truths": resolve_cones(scn, road)} for _ in range(max(args.repeats, 1))]
    else:
        road, runs = run_carla(scn, cfg, geom, max(args.repeats, 1), args.settle_ticks,
                               args.allow_synthetic_cones, args.hold_seconds, args.view)

    # ------------------------------------------------------------ repeatability
    def cone_xy(run):
        if run["records"]:
            return np.array([[rec.actual_carla["x"], rec.actual_carla["y"]] if rec.actual_carla
                             else [rec.requested_carla["x"], rec.requested_carla["y"]] for rec in run["records"]])
        return np.array([[c.x, c.y] for c in run["truths"]])

    base = cone_xy(runs[0])
    cone_dev = max((float(np.max(np.abs(cone_xy(r) - base))) for r in runs[1:]), default=0.0)
    ego_dev = max((float(np.hypot(r["ego"].x - runs[0]["ego"].x, r["ego"].y - runs[0]["ego"].y))
                   for r in runs[1:]), default=0.0)
    repeat_ok = cone_dev < 1e-3
    print(f"[scenario] repeatability over {len(runs)} runs: max cone deviation {cone_dev * 1000:.3f} mm "
          f"({'OK' if repeat_ok else 'MISMATCH'}), settled ego deviation {ego_dev * 100:.2f} cm")

    # ------------------------------------------- observations through provider
    provider = GroundTruthConeProvider(runs[0]["truths"])
    observations = provider.get_cones(runs[0]["ego"])
    scene = {
        "scenario": scn.name, "source_file": scn.source, "mode": "offline" if args.offline else "carla",
        "map": scn.map_name, "seed": scn.seed,
        "frames": {"map": "right-handed: x_map=x_carla, y_map=-y_carla, yaw_map=-yaw_carla (rad)",
                   "road": "s along ego-lane centre, d left-positive (m)"},
        "road": road.to_dict(),
        "corridors": [vars(c) for c in scn.corridors], "goal": vars(scn.goal),
        "ego_start_map": vars(runs[0]["ego"]),
        "observations_map": [o.to_dict() for o in observations],
        "cone_registry": [rec.to_dict() for rec in runs[0]["records"]] if runs[0]["records"] else None,
        "repeatability": {"runs": len(runs), "max_cone_dev_m": cone_dev, "max_ego_dev_m": ego_dev},
    }
    (out_dir / "scene.json").write_text(json.dumps(scene, indent=2, default=float))
    png = plot_scene(road, observations, runs[0]["ego"], geom, scn.corridors, scn.goal,
                     title=f"{scn.name} ({len(observations)} cones)", out_path=out_dir / "scene.png",
                     margin_m=margin, synthetic=args.offline)

    print("[scenario] cone ground truth (map frame | road frame):")
    for o in observations:
        s, d = road.reference.to_frenet(o.x, o.y)
        print(f"  {o.cone_id:>4} {o.state:8s} x={o.x:9.3f} y={o.y:9.3f} | s={s:6.2f} d={d:+.2f}")
    print(f"[scenario] scene -> {out_dir / 'scene.json'}\n[scenario] plot  -> {png}")
    print(f"[scenario] RESULT: {'PASS' if repeat_ok else 'FAIL'}")
    return 0 if repeat_ok else 1


if __name__ == "__main__":
    sys.exit(main())
