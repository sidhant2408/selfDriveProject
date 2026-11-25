"""Phase 3+4: closed-loop episode (ground-truth cones -> planner -> MPC -> vehicle).

CARLA mode (default): the MPC drives the CARLA vehicle through the calibrated
actuator map; no autopilot / Traffic Manager.
Offline mode (--offline): same loop on a kinematic-bicycle plant and a synthetic
straight road (tagged SYNTHETIC; validates integration, not CARLA dynamics).

Usage:
    python scripts/run_episode.py scenarios/partial_obstruction.yaml [--realtime] [--view chase|topdown|none]
    python scripts/run_episode.py scenarios/full_closure.yaml --offline
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

from cone_navigation.config import PROJECT_ROOT, load_config
from cone_navigation.ground_truth import GroundTruthConeProvider
from cone_navigation.mpc import KinematicMPC, MPCConfig
from cone_navigation.planner import LocalPlanner, PlannerConfig
from cone_navigation.plant import KinematicPlant
from cone_navigation.road import synthetic_straight_road
from cone_navigation.runner import expected_outcome, run_episode
from cone_navigation.scenario import ScenarioError, load_scenario, resolve_cones, resolve_ego_start, validate_geometry
from cone_navigation.state import VehicleGeometry
from cone_navigation.visualization import plot_episode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenario", nargs="?", default=str(PROJECT_ROOT / "scenarios" / "partial_obstruction.yaml"))
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--realtime", action="store_true", help="CARLA: pace to wall clock for watching")
    ap.add_argument("--view", choices=["chase", "topdown", "none"], default="chase")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--out", default=str(PROJECT_ROOT / "outputs"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    scn = load_scenario(args.scenario)
    geom = VehicleGeometry.from_config(cfg["vehicle"])
    dt = float(cfg["simulation"]["fixed_delta_seconds"])
    max_s = args.max_seconds or float(cfg["simulation"].get("max_episode_seconds", 40))
    plan_every = int(cfg["planner"].get("update_every_n_ticks", 4))
    pcfg = PlannerConfig.from_config(cfg)
    mcfg = MPCConfig.from_config(cfg)
    out_dir = Path(args.out) / f"episode_{scn.name}_{'offline' if args.offline else 'carla'}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[episode] {scn.name}: {len(scn.cones)} cones, {'OFFLINE kinematic plant' if args.offline else 'CARLA'}; "
          f"margin {pcfg.cone_margin_m} m, target {pcfg.target_speed_mps} m/s, MPC N={mcfg.horizon_steps} dt={mcfg.dt}")

    mpc = KinematicMPC(mcfg)
    planner = LocalPlanner(pcfg, geom)

    def progress(k, ego, plan, res, row):
        if k % 40 == 0:
            print(f"  t={row['t_s']:5.1f}s s={row['s']:6.2f} d={row['d']:+.2f} v={row['speed_mps']:4.2f} "
                  f"{plan.decision:4s} a={res.accel:+.2f} delta={res.delta:+.3f} "
                  f"clear={row['clearance_m']:.2f} mpc={'ok' if res.ok else 'FAIL'} {row['mpc_ms'] if row['mpc_ms'] == row['mpc_ms'] else 0:.0f}ms")

    if args.offline:
        road = synthetic_straight_road(scn.route.length_m, scn.route.sample_spacing_m)
        problems = validate_geometry(scn, road, geom)
        if problems:
            raise ScenarioError(problems, scn.source)
        truth = resolve_cones(scn, road)
        plant = KinematicPlant(resolve_ego_start(scn, road), mcfg.wheelbase_m, mcfg.rear_axle_offset_m, dt)
        result = run_episode(plant, GroundTruthConeProvider(truth), planner, mpc, road, scn, geom, truth,
                             dt, max_s, plan_every, on_tick=progress)
    else:
        import carla
        from cone_navigation.actuation import ActuatorMapper
        from cone_navigation.carla_plant import CarlaPlant
        from cone_navigation.carla_session import CarlaSession
        from cone_navigation.carla_world import build_road_geometry, read_back_cones, spawn_cones, spawn_ego

        with CarlaSession(cfg["simulation"], map_name=scn.map_name, realtime=args.realtime) as sess:
            road = build_road_geometry(sess.map, scn.route)
            problems = validate_geometry(scn, road, geom)
            if problems:
                raise ScenarioError(problems, scn.source)
            truth = resolve_cones(scn, road)
            ego = spawn_ego(sess, cfg["vehicle"]["blueprint"], resolve_ego_start(scn, road),
                            float(road.reference.z_at(scn.ego.s)), scn.ego.spawn_z_offset_m)
            records = spawn_cones(sess, truth, scn.cone_blueprint, road)
            for _ in range(20):
                ego.apply_control(carla.VehicleControl(brake=1.0)); sess.tick()
            read_back_cones(sess, records)
            if args.view == "topdown":
                mx, my = road.reference.to_world(road.reference.length / 2, 1.75)[:2]
                sess.set_spectator_topdown(mx, -my, 60.0)
            plant = CarlaPlant(sess, ego, ActuatorMapper(cfg["actuation"], cfg["vehicle"]),
                               view="chase" if args.view == "chase" else None)
            result = run_episode(plant, GroundTruthConeProvider(truth), planner, mpc, road, scn, geom, truth,
                                 dt, max_s, plan_every, on_tick=progress)
            plant.ego.apply_control(carla.VehicleControl(brake=1.0))
            if sess.tick_timeouts:
                result.summary["tick_timeouts"] = sess.tick_timeouts

    expected = expected_outcome(scn.ground_truth_label)   # evaluation only
    s = result.summary
    s.update({"scenario": scn.name, "mode": "offline" if args.offline else "carla",
              "expected_outcome": expected, "passed": result.outcome == expected})
    with (out_dir / "timeseries.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(result.rows[0].keys()))
        w.writeheader(); w.writerows(result.rows)
    (out_dir / "summary.json").write_text(json.dumps(s, indent=2, default=str))
    png = plot_episode(road, truth, result.rows, result.plans, geom, scn.corridors, scn.goal, pcfg.cone_margin_m,
                       f"{scn.name} - {s['mode']} - outcome {result.outcome} (expected {expected})"
                       + ("   [SYNTHETIC road, kinematic plant]" if args.offline else ""),
                       out_dir / "episode.png")

    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) and math.isfinite(v) else str(v)
    print("\n[episode] summary:")
    for k in ("outcome", "expected_outcome", "duration_s", "final_s", "final_speed_mps", "min_true_clearance_m",
              "max_corridor_violation_m", "cross_track_rms_m", "cross_track_max_m", "max_speed_mps",
              "peak_decel_measured_mps2", "mpc_solves", "mpc_failures", "mpc_ms_mean", "mpc_ms_p95",
              "plan_ms_mean", "plan_ms_max", "sim_collision_events"):
        print(f"  {k:26s} {fmt(s.get(k))}")
    print(f"[episode] outputs -> {out_dir}\n[episode] plot    -> {png}")
    print(f"[episode] RESULT: {'PASS' if s['passed'] else 'FAIL'} ({result.outcome}, expected {expected})")
    return 0 if s["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
