"""Measure the CARLA actuator response of the ego vehicle (Phase 4 prerequisite).

Each trial respawns the car at the scenario anchor on the straight test road:
  * throttle trials: constant throttle from rest -> acceleration vs throttle
  * brake trials:    reach ~4 m/s, then constant brake -> deceleration vs brake
  * coast trial:     reach ~4 m/s, then no input -> natural deceleration
  * steer trials:    hold ~3 m/s, small constant steer -> effective wheel angle
Results are printed and saved to outputs/calibration_<time>/calibration.json.
Copy the suggested values into configs/default.yaml (section `actuation`).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from cone_navigation.config import PROJECT_ROOT, load_config
from cone_navigation.frames import wrap_angle
from cone_navigation.scenario import load_scenario, resolve_ego_start


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    ap.add_argument("--scenario", default=str(PROJECT_ROOT / "scenarios" / "empty_lane.yaml"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    scn = load_scenario(args.scenario)
    veh = cfg["vehicle"]
    dt = float(cfg["simulation"]["fixed_delta_seconds"])

    import carla
    from cone_navigation.carla_session import CarlaSession
    from cone_navigation.carla_world import build_road_geometry, read_ego_state, spawn_ego

    out: dict = {"throttle": [], "brake": [], "coast": None, "steer": []}
    with CarlaSession(cfg["simulation"], map_name=scn.map_name) as sess:
        road = build_road_geometry(sess.map, scn.route)
        start = resolve_ego_start(scn, road)

        def trial(control_fn, seconds):
            ego = spawn_ego(sess, veh["blueprint"], start, float(road.reference.z_at(0.0)), 0.3)
            for _ in range(20):                            # settle
                ego.apply_control(carla.VehicleControl(brake=1.0)); sess.tick()
            log = []
            for k in range(int(seconds / dt)):
                st = read_ego_state(sess, ego)
                thr, steer, brk, phase = control_fn(k * dt, st)
                ego.apply_control(carla.VehicleControl(throttle=thr, steer=steer, brake=brk))
                sess.tick()
                log.append((k * dt, read_ego_state(sess, ego), phase))
            sess.destroy_actors()
            return log

        def accel_fit(log, phase, v_lo, v_hi):
            pts = [(t, s.speed_mps) for t, s, p in log if p == phase and v_lo <= s.speed_mps <= v_hi]
            if len(pts) < 5:
                return float("nan")
            t, v = np.array(pts).T
            return float(np.polyfit(t, v, 1)[0])

        for thr in (0.2, 0.3, 0.4, 0.5, 0.6):
            log = trial(lambda t, s, thr=thr: (thr, 0.0, 0.0, "go"), 5.0)
            a = accel_fit(log, "go", 0.3, 4.5)
            out["throttle"].append({"throttle": thr, "accel_mps2": a})
            print(f"throttle {thr:.1f}: accel {a:+.2f} m/s^2 (0.3..4.5 m/s)")

        def brake_fn(level):
            state = {"braking": False}

            def f(t, s):
                if not state["braking"] and s.speed_mps >= 4.0:
                    state["braking"] = True
                if state["braking"]:
                    return 0.0, 0.0, level, "brake" if level > 0 else "coast"
                return 0.6, 0.0, 0.0, "spinup"
            return f

        for brk in (0.1, 0.2, 0.3, 0.5):
            log = trial(brake_fn(brk), 8.0)
            a = accel_fit(log, "brake", 0.5, 4.2)
            out["brake"].append({"brake": brk, "accel_mps2": a})
            print(f"brake    {brk:.1f}: accel {a:+.2f} m/s^2")
        log = trial(brake_fn(0.0), 9.0)
        out["coast"] = accel_fit(log, "coast", 0.5, 4.2)
        print(f"coast        : accel {out['coast']:+.2f} m/s^2")

        L = float(veh["wheelbase_m"])
        max_wheel = math.radians(float(veh["max_wheel_steer_deg"]))
        for steer in (0.05, 0.10, 0.15):
            def f(t, s, steer=steer):
                thr = float(np.clip(0.35 + 0.3 * (3.0 - s.speed_mps), 0.0, 0.7))  # hold ~3 m/s
                return thr, (steer if t >= 4.0 else 0.0), 0.0, ("steer" if t >= 5.0 else "spin")
            log = trial(f, 6.5)
            seg = [(t, s) for t, s, p in log if p == "steer"]
            rates = [wrap_angle(b.yaw - a.yaw) / dt for (_, a), (_, b) in zip(seg, seg[1:])]
            v = float(np.mean([s.speed_mps for _, s in seg]))
            yr = float(np.mean(rates))
            delta_eff = math.atan(abs(yr) * L / max(v, 0.1))
            gain = delta_eff / (steer * max_wheel)
            out["steer"].append({"steer": steer, "speed": v, "yaw_rate_map": yr,
                                 "effective_wheel_rad": delta_eff, "gain_vs_linear": gain})
            print(f"steer   {steer:.2f}: v={v:.2f} yaw rate {yr:+.4f} rad/s -> wheel {math.degrees(delta_eff):.2f} deg "
                  f"(linear model {math.degrees(steer * max_wheel):.2f} deg, gain {gain:.2f})")

    # --- suggested linear maps -------------------------------------------------
    thr = np.array([[d["throttle"], d["accel_mps2"]] for d in out["throttle"]])
    k_thr, a0 = np.polyfit(thr[:, 0], thr[:, 1], 1)          # a = k*thr + a0
    brk = np.array([[d["brake"], d["accel_mps2"] - out["coast"]] for d in out["brake"]])
    k_brk = float(np.sum(brk[:, 0] * -brk[:, 1]) / np.sum(brk[:, 0] ** 2))  # decel = k*brake (through 0)
    out["suggested"] = {
        "throttle_accel_gain": float(k_thr), "throttle_accel_offset": float(a0),
        "brake_decel_gain": k_brk, "coast_accel_mps2": float(out["coast"]),
        "steer_gain": float(np.mean([d["gain_vs_linear"] for d in out["steer"]])),
    }
    print("\nsuggested `actuation` config:", json.dumps(out["suggested"], indent=2))
    d = PROJECT_ROOT / "outputs" / f"calibration_{time.strftime('%Y%m%d_%H%M%S')}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "calibration.json").write_text(json.dumps(out, indent=2))
    print(f"saved -> {d / 'calibration.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
