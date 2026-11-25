"""Phase 0 smoke test: connect, sync mode, spawn ego, open-loop drive 10 s, log, clean up.

The controller here is a fixed open-loop schedule (no autopilot, no Traffic
Manager). Per tick the order is:
    apply_control(u_k)  ->  world.tick()  ->  read snapshot of the NEW frame
so logged row k is the state that resulted from command u_k.

Usage:
    python scripts/smoke_test.py [--config configs/default.yaml]
                                 [--scenario scenarios/partial_obstruction.yaml]
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
from cone_navigation.frames import wrap_angle
from cone_navigation.scenario import load_scenario, resolve_ego_start


def schedule(t: float) -> tuple[float, float, float]:
    """(throttle, steer, brake) as a function of sim time. Low speed by design."""
    if t < 1.0:
        return 0.0, 0.0, 1.0          # settle on the ground, held
    if t < 4.0:
        return 0.35, 0.0, 0.0         # straight acceleration
    if t < 5.5:
        return 0.25, +0.10, 0.0       # small positive steer (sign check)
    if t < 7.0:
        return 0.25, -0.10, 0.0       # small negative steer (sign check, re-centre)
    return 0.0, 0.0, 0.6              # brake to stop


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    ap.add_argument("--scenario", default=str(PROJECT_ROOT / "scenarios" / "partial_obstruction.yaml"),
                    help="only its map and route anchor are used")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "outputs"))
    ap.add_argument("--realtime", action="store_true", help="pace ticks to wall clock so you can watch")
    ap.add_argument("--view", choices=["chase", "topdown"], default="chase", help="spectator camera")
    ap.add_argument("--with-cones", action="store_true",
                    help="also place the scenario's cones (visual context only; the car stops before them)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    scn = load_scenario(args.scenario)
    sim, veh = cfg["simulation"], cfg["vehicle"]
    dt = float(sim["fixed_delta_seconds"])
    duration = float(cfg.get("smoke_test", {}).get("duration_s", 10.0))
    n_ticks = int(round(duration / dt))
    out_dir = Path(args.out) / f"smoke_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # carla imports only here so --help works without the simulator package.
    import carla
    from cone_navigation.carla_session import CarlaSession
    from cone_navigation.carla_world import build_road_geometry, read_ego_state, spawn_cones, spawn_ego
    from cone_navigation.scenario import resolve_cones
    from cone_navigation.visualization import plot_smoke_log

    rows: list[dict] = []
    with CarlaSession(sim, map_name=scn.map_name, realtime=args.realtime) as sess:
        print(f"[smoke] CARLA client {sess.versions['client']} / server {sess.versions['server']}, "
              f"map {sess.map.name}, sync dt={dt}s")
        road = build_road_geometry(sess.map, scn.route)
        start = resolve_ego_start(scn, road)
        ego = spawn_ego(sess, veh["blueprint"], start, float(road.reference.z_at(scn.ego.s)),
                        scn.ego.spawn_z_offset_m)
        print(f"[smoke] spawned {veh['blueprint']} id={ego.id} on road {road.meta['road_id']} "
              f"lane {road.meta['lane_id']}")
        if args.with_cones:
            spawn_cones(sess, resolve_cones(scn, road), scn.cone_blueprint, road)
        sess.tick()
        mid_x, mid_y = road.reference.to_world(15.0, 0.0)[:2]
        sess.set_spectator_topdown(mid_x, -mid_y, 35.0)

        def update_view():
            if args.view == "chase":
                sess.follow_spectator(ego)

        update_view()

        prev = read_ego_state(sess, ego)
        t0 = prev.timestamp_s
        for k in range(n_ticks):
            t = k * dt
            thr, steer, brk = schedule(t)
            ego.apply_control(carla.VehicleControl(throttle=thr, steer=steer, brake=brk,
                                                   hand_brake=False, reverse=False))
            frame = sess.tick()
            update_view()
            st = read_ego_state(sess, ego)
            s, d = road.reference.to_frenet(st.x, st.y)
            rows.append({
                "frame": frame, "t_s": round(st.timestamp_s - t0, 6),
                "x_map": st.x, "y_map": st.y, "yaw_map_rad": st.yaw,
                "x_carla": st.x, "y_carla": -st.y, "yaw_carla_deg": -math.degrees(st.yaw),
                "s": s, "d": d, "speed_mps": st.speed_mps,
                "yaw_rate_map_rps": wrap_angle(st.yaw - prev.yaw) / dt,
                "throttle": thr, "steer": steer, "brake": brk,
            })
            prev = st
            if k % 20 == 0:
                print(f"  t={rows[-1]['t_s']:5.2f}s frame={frame} x={st.x:8.2f} y_map={st.y:8.2f} "
                      f"yaw={math.degrees(st.yaw):7.2f}deg v={st.speed_mps:5.2f}m/s s={s:6.2f} d={d:+.2f}")
    print("[smoke] actors destroyed, original world settings restored")

    # ------------------------------------------------------------------ checks
    def mean_rate(t_lo, t_hi):
        v = [r["yaw_rate_map_rps"] for r in rows if t_lo <= r["t_s"] < t_hi]
        return sum(v) / max(len(v), 1)

    sign = int(veh.get("steer_sign_map", -1))
    r_pos, r_neg = mean_rate(4.5, 5.5), mean_rate(6.0, 7.0)
    frames = [r["frame"] for r in rows]
    dist = sum(math.hypot(b["x_map"] - a["x_map"], b["y_map"] - a["y_map"]) for a, b in zip(rows, rows[1:]))
    vmax = max(r["speed_mps"] for r in rows)
    checks = [
        ("synchronous: consecutive frame ids", all(b - a == 1 for a, b in zip(frames, frames[1:])),
         f"{frames[0]}..{frames[-1]}"),
        ("fixed step: sim time == n*dt", abs(rows[-1]["t_s"] - n_ticks * dt) < 1e-3,
         f"{rows[-1]['t_s']:.4f}s vs {n_ticks * dt:.4f}s"),
        ("vehicle moved > 3 m", dist > 3.0, f"{dist:.2f} m"),
        ("low speed (< 8 m/s)", vmax < 8.0, f"max {vmax:.2f} m/s"),
        ("steer +0.1 -> yaw-rate sign == steer_sign_map", r_pos * sign > 0.005,
         f"mean {r_pos:+.4f} rad/s, expected sign {sign:+d}"),
        ("steer -0.1 -> opposite yaw-rate sign", r_neg * sign < -0.005, f"mean {r_neg:+.4f} rad/s"),
        ("stopped at end (< 0.3 m/s)", abs(rows[-1]["speed_mps"]) < 0.3, f"{rows[-1]['speed_mps']:.2f} m/s"),
    ]

    csv_path = out_dir / "state_log.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    png = plot_smoke_log(rows, out_dir / "smoke_test.png", f"Smoke test - {scn.map_name}, {veh['blueprint']}")
    ok = all(c[1] for c in checks)
    (out_dir / "summary.json").write_text(json.dumps(
        {"passed": ok, "checks": [{"name": n, "passed": p, "detail": d} for n, p, d in checks],
         "distance_m": dist, "max_speed_mps": vmax, "ticks": n_ticks, "dt": dt}, indent=2))

    print("\n[smoke] checks:")
    for name, passed, detail in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    print(f"[smoke] log  -> {csv_path}\n[smoke] plot -> {png}")
    print(f"[smoke] RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
