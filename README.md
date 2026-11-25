# cone_navigation: ground-truth-first cone-aware driving in CARLA

Status: **Phases 0, 1, 3 and 4 run end-to-end.** Verified on this machine on
2026-11-23 against CARLA 1.0 (UE 5.8 package, client 1.0 = server 1.0),
Windows 11, Python 3.15.2, CasADi 3.8.2 / IPOPT.

- **Phase 2 is not built yet.** There is no PARTIAL/DIVERSION/CLOSURE classifier. The planner decides pass versus stop purely from geometry: does a collision-free path exist inside the permitted corridors?
- **Phase 5 is not built yet** (noise and adaptive margins).

## System diagram and data flow

```
 scenarios/*.yaml ──load+validate──► Scenario (road frame s,d; corridors; goal)
                                          │
 CARLA map ──build_road_geometry──► RoadGeometry (ref. line in MAP frame, lanes)
                                          │ resolve (s,d) -> map -> CARLA
                                          ▼
              carla_world.spawn_* ──► CARLA actors (ego + cone props)
                                          │ registry kept at spawn time
                                          ▼
   GROUND TRUTH ENTERS HERE ──► GroundTruthConeProvider.get_cones() ─► [ConeObservation] (map frame)
                                          │
   (Phase 5) NoiseAdapter wraps the provider, same signature
   (Phase 6) camera/LiDAR fusion implements the same signature
                                          ▼
   Phase 2  pattern_detector.classify(observations, corridors) -> PatternResult
   Phase 3  planner.plan(ego_state, pattern, observations, road) -> [ReferencePoint]
   Phase 4  mpc.solve(ego_state, reference) -> (a, delta)   ◄── MPC ATTACHES HERE
            carla_control.to_carla(a, delta) -> VehicleControl ─► ego.apply_control
                                          │
   loop:  apply_control(u_k) -> world.tick() -> snapshot of new frame -> EgoState
```

Only `carla_*.py` import `carla`. Everything downstream of the provider works on
dataclasses, so all of it is testable without the simulator.

## Frames (also in `src/cone_navigation/frames.py`)

| Frame | Definition |
|---|---|
| CARLA world | left-handed: x fwd, **y right**, z up; yaw in degrees, clockwise from above |
| Map (all algorithms) | right-handed: `x_m = x_c`, `y_m = -y_c`, `yaw_m = -radians(yaw_c)` |
| Road `(s, d)` | `s` = arc length along the ego-lane centre, `d` = lateral, **left positive** |
| Ego body | x forward, y left; origin = CARLA actor origin; bbox centre at `x = -0.0685 m` |

Measured facts used in the config (`configs/default.yaml`):

| Item | Value | How it was measured |
|---|---|---|
| Ego | `vehicle.tesla.model3`: 4.694 × 2.088 m, wheelbase 2.875 m, max wheel steer 35° | bbox + `VehiclePhysicsControl.wheels` (wheel locations are in **world metres** in 1.0) |
| Steering sign | `steer=+0.1` gives map yaw rate -0.049 rad/s (a **right** turn) → `steer_sign_map: -1` | smoke test |
| Cone | `static.prop.constructioncone`: radius 0.239 m, height 0.712 m, origin at base | spawned bbox |
| Cone physics | static props **do not move** under gravity or impact, and the collision sensor **does** fire on them | spawn and drive-into test |
| Test road | Town10HD_Opt, road 7, lane -2, anchor (-24.34, -57.79): 91 m straight, same-direction lane on the left | waypoint scan |

`vehicle.lincoln.mkz` reports `max_steer_angle = 70°`, which looks wrong, so it is not used.

## Setup (Windows PowerShell)

The existing `C:\UMICH_PYTHON\carla-env` venv (Python 3.15) already contains the
matching `carla 1.0` client. To recreate it, install the wheel that ships with
the server package, never one from PyPI:

```powershell
cd C:\UMICH_PYTHON
py -3.15 -m venv carla-env
.\carla-env\Scripts\python.exe -m pip install CARLA\Carla-1.0-Win64-Shipping\PythonAPI\carla\dist\carla-1.0-cp315-cp315-win_amd64.whl
cd cone_navigation
..\carla-env\Scripts\python.exe -m pip install -e ".[dev]"
```

Start the server (in its own terminal; `-RenderOffScreen` gives no window):

```powershell
C:\UMICH_PYTHON\CARLA\Carla-1.0-Win64-Shipping\CarlaUnreal.exe -RenderOffScreen -quality-level=Low
```

Leave out `-RenderOffScreen` to watch. The scripts move the spectator to a
top-down view of the scene.

## Commands

```powershell
cd C:\UMICH_PYTHON\cone_navigation
$py = "..\carla-env\Scripts\python.exe"
& $py -m pytest -q                                   # no CARLA needed
& $py scripts\run_scenario.py --offline              # no CARLA needed (SYNTHETIC road)
& $py scripts\smoke_test.py                          # Phase 0 (needs server)
& $py scripts\run_scenario.py scenarios\partial_obstruction.yaml   # Phase 1 (needs server)
& $py scripts\run_scenario.py scenarios\tapered_diversion.yaml
& $py scripts\run_scenario.py scenarios\full_closure.yaml
```

| Scenario | File | Cones | Permitted corridors | Correct behaviour |
|---|---|---|---|---|
| A partial obstruction | `partial_obstruction.yaml` | 5 along the right lane edge | ego lane | pass, staying in the ego lane |
| B tapered diversion | `tapered_diversion.yaml` | 6-cone taper + 4 along the lane line | ego lane + left lane from s = 10 m | shift left, optionally return after s = 50 m |
| C full closure | `full_closure.yaml` | 9 across both lanes at s = 40 m (1 fallen) | ego lane + left lane | stop before s = 40 m |

### Closed-loop driving (Phases 3 + 4)

```powershell
& $py scripts\run_episode.py scenarios\empty_lane.yaml            # MPC lane-centre tracking
& $py scripts\run_episode.py scenarios\partial_obstruction.yaml   # A: pass in lane
& $py scripts\run_episode.py scenarios\tapered_diversion.yaml     # B: divert left and return
& $py scripts\run_episode.py scenarios\full_closure.yaml          # C: stop before the barrier
& $py scripts\run_episode.py scenarios\full_closure.yaml --offline   # no CARLA: kinematic plant
& $py scripts\run_episode.py scenarios\tapered_diversion.yaml --realtime --view chase   # watch it
& $py scripts\calibrate_actuators.py                               # re-measure throttle/brake/steer maps
```

Each episode writes `outputs/episode_<scenario>_<mode>_<time>/`:
- `timeseries.csv`: per tick, 20 Hz
- `summary.json`: the metrics below
- `episode.png`: road-frame trajectory over the true cones, plus speed and command plots

**Results (actual runs, 2026-11-23, CARLA mode, final code):**

| Scenario | Outcome (expected) | Min true clearance | Cross-track RMS / max | Peak decel | MPC failures | MPC p95 | Plan max |
|---|---|---|---|---|---|---|---|
| empty_lane | GOAL (GOAL) | n/a | 1 / 3 mm | 0.12 m/s² | 0 / 243 | 4.1 ms | 9 ms |
| A partial | GOAL (GOAL) | 0.44 m | 3 / 10 mm | 0.12 m/s² | 0 | 3.4 ms | 26 ms |
| B diversion | GOAL (GOAL) | 0.47 m | 4 / 15 mm | 0.12 m/s² | 0 | 3.5 ms | 261 ms |
| C closure | STOPPED at s=35.7 m (STOPPED) | 1.78 m | 0 / 1 mm | 2.07 m/s² | 0 | 4.5 ms | 80 ms |

All runs had zero corridor violations and no collision-sensor events. The same four scenarios also pass offline on the kinematic plant, which is what `tests/test_closed_loop_offline.py` checks. Each result is a single run with no noise, so it is not a statistical safety claim.

**How it works:**
- **Planner** ([planner.py](src/cone_navigation/planner.py), 5 Hz)
  - Samples quintic lateral-offset paths: target offset × transition length × start delay, with or without a return to the lane centre.
  - Rejects any path where the **full ego rectangle** (bbox + 5 cm), checked every 0.5 m, touches a cone inflated by 0.40 m, or has a corner outside the permitted corridors, or exceeds the curvature limit.
  - Picks the cheapest remaining path. The cost combines linear deviation from the lane centre, curvature, and consistency with the committed path.
  - Keeps the committed path while it stays safe. Without this, replanning postpones a manoeuvre until it becomes infeasible; that was a real bug found in CARLA.
  - If no path is safe, it outputs a stop point 1.5 m (ego centre) before the first conflict, with speed profile `v = sqrt(2 a (s_stop - s))`.
- **MPC** ([mpc.py](src/cone_navigation/mpc.py), 10 Hz)
  - Kinematic bicycle model with a rear-axle state. The cost is on the actor-origin position, so it matches the plan exactly.
  - N = 20 steps at dt = 0.1 s (2 s horizon), solved with CasADi/IPOPT and warm-started.
  - Hard limits on steering, steering rate, acceleration/braking and speed.
  - Brakes at maximum if the solver fails or its inputs are not finite.
- **Actuation** ([actuation.py](src/cone_navigation/actuation.py))
  - Inverts measured throttle and brake lookup tables, plus speed-error feedback.
  - Steer = `steer_sign_map · δ / (0.77 · 35°)`, rate-limited to 0.05 per tick.
  - Holds the brake at standstill.

**Calibration** (measured, Tesla Model 3, 0.3–4.5 m/s):

| Input | Response |
|---|---|
| throttle 0.2 / 0.3 / 0.4 / 0.5 / 0.6 | +0.29 / +0.64 / +1.13 / +1.78 / +2.57 m/s² |
| brake 0.1 / 0.2 / 0.3 / 0.5 | −1.31 / −2.45 / −3.60 / −5.93 m/s² |
| coasting | −0.18 m/s² |
| steer | effective wheel angle = 0.77 × steer × 35° |

### Watching the simulation

Start the server **without** `-RenderOffScreen`, then run the scripts with the viewing options:

```powershell
C:\UMICH_PYTHON\CARLA\Carla-1.0-Win64-Shipping\CarlaUnreal.exe -windowed -ResX=1280 -ResY=720 -quality-level=Low
& $py scripts\smoke_test.py --realtime --view chase --with-cones        # car drives ~14 m toward the cones
& $py scripts\run_scenario.py --hold-seconds 30 --view chase            # static scene stays up 30 s
```

Without these options the scripts tick as fast as the server allows, about 5× real time. In synchronous mode the window only updates while a script is ticking; when a script exits, the server returns to normal asynchronous mode.

Outputs go to `outputs/<name>_<timestamp>/`:
- smoke test: `state_log.csv`, `smoke_test.png`, `summary.json`
- scenario: `scene.json` (road, corridors, cone registry, observations), `scene.png`

## Example output (actual output from this machine, 2026-11-23; abridged)

```
[smoke] checks:
  [PASS] synchronous: consecutive frame ids: 2218984..2219183
  [PASS] fixed step: sim time == n*dt: 10.0000s vs 10.0000s
  [PASS] vehicle moved > 3 m: 14.09 m
  [PASS] low speed (< 8 m/s): max 3.67 m/s
  [PASS] steer +0.1 -> yaw-rate sign == steer_sign_map: mean -0.0493 rad/s, expected sign -1
  [PASS] steer -0.1 -> opposite yaw-rate sign: mean +0.0577 rad/s
  [PASS] stopped at end (< 0.3 m/s): 0.00 m/s
[smoke] RESULT: PASS

[scenario] reference line 85.1 m on road 7; lanes: ego(id -2, d=+0.00, same), left_1(id 1, d=+3.50, same),
           left_2(id -1, d=+7.00, opposite), left_3(id -2, d=+10.50, opposite)
[scenario] run 0: 5/5 cones spawned as visible props, max cone drift after 20 ticks = 0.00 cm, ego at s,d = (-0.001, -0.004)
[scenario] run 1: 5/5 cones spawned as visible props, ...
[scenario] repeatability over 2 runs: max cone deviation 0.000 mm (OK), settled ego deviation 0.00 cm
[scenario] RESULT: PASS
```

Success criteria:
- **Phase 0:** all seven smoke checks pass. This includes consecutive frame IDs (synchronous mode really on) and the steering sign agreeing with `steer_sign_map`.
- **Phase 1:** every cone spawns as a visible prop, cones show zero drift after settling, re-spawned cones agree to < 1 mm, and the plot shows cones, lanes, corridor and goal.

## Troubleshooting

- **`Cannot reach a CARLA server`**: start `CarlaUnreal.exe` and check that port 2000 is free (`Test-NetConnection 127.0.0.1 -Port 2000`). Use `-carla-rpc-port=N` and set `simulation.port` to match.
- **`client X != server Y`**: install the client wheel from *this* package's `PythonAPI\carla\dist`. 0.9.x, 0.10 and 1.0 clients are not interchangeable.
- **`Map ... is not available`**: this package only has the cooked maps `Town10HD_Opt`, `EmptyMap`, `Town12` and `Town15`. Other towns need the AdditionalMaps content pack.
- **`Blueprint ... not found`**: the error lists similar IDs. Cones in this build are `static.prop.constructioncone`, `static.prop.trafficcone01` and `static.prop.trafficcone02`. `--allow-synthetic-cones` continues with invisible records that are flagged `visual: false`.
- **Server looks frozen after a crash**: it may be stuck in synchronous mode. Run any script once (they restore settings on exit), or restart the server.
- **Exit code is non-zero although output says PASS**: the CARLA skills note that the client can segfault at interpreter teardown. Trust the `RESULT:` line.

## Known issues / not yet verified

- **Intermittent tick timeout (unresolved).** Early on, 3 of about 6 `run_scenario` runs hit a 30 s `world.tick()` timeout on the first run in the process. I could not reproduce it afterwards: 9 consecutive passes, including smoke-then-scenario sequences.
  - `CarlaSession.tick()` now logs the server frame on a timeout. It continues if the server did advance, otherwise it retries once.
  - Observed once (scenario C, windowed server, `--hold-seconds 30`): the server frame did **not** advance for 10 s, and the single retry succeeded, so the run passed. The server stalls on a step; the reply is not being lost. Root cause still unknown.
- **Fallen cones** (`Rotation(pitch=-90, yaw)`): checked in a camera image of scenario C on 2026-11-23. The cone lies flat on the road along its yaw, in its footprint position.
- **Town10HD has parked vehicles baked into the map**: they are environment objects, not actors, so they are invisible to the ground-truth pipeline. Camera frames show a van and taxis parked to the right of the solid edge line, outside the ego lane, at least as far as s ≈ 60 m. They are still not modelled. If a future route or scenario brings the car near the kerb, hide them (`world.enable_environment_objects`) or add them as obstacles.
- Static cone props act as **rigid posts**: a car hitting one stops rather than knocking it over. Safety metrics will use geometric clearance, not physics.
- **Cross-track error understates tracking quality.** It is measured against the *current* plan, which is re-anchored near the car on every replan. The true-clearance and corridor-violation metrics are the meaningful safety numbers.
- **Planning time:** about 600 candidates in B take up to about 260 ms per plan. That is harmless in synchronous mode, but with `--realtime` the window stutters during those plans. An early-exit collision check would help.
- **Steering steps:** small steps in the wheel-angle command appear when the planner switches paths. The rate constraint bounds them (≤ 0.6 rad/s), but they are not perfectly smooth.
- **Calibration** is single-vehicle, low-speed and assumes a flat road. The kinematic model ignores tyre slip, which is fine at 3 m/s and not validated above about 4.5 m/s.
- **Straight-road assumption:** the planner works in (s, d) assuming a nearly straight reference line, and the adapter rejects bends over 2°.

## Next step: Phase 2 classifier, then Phase 5

The planner already handles pass versus stop geometrically. Phase 2 adds the interpretable label and reason codes on top of it, and Phase 5 then adds noisy observations with fixed versus adaptive margins (`planner.base_clearance_m` is the single margin knob).

### Phase 2 geometric passability (original plan)

Add `pattern_detector.py`, taking `[ConeObservation]`, `RoadGeometry` and the corridors and returning a `PatternResult`:
1. Project cones to `(s, d)` and inflate the footprints by the margin.
2. Cluster cones along `s`.
3. For each `s`-slice, compute the free lateral intervals inside each permitted corridor.
4. Label the scene PARTIAL / DIVERSION / CLOSURE / UNKNOWN, with reason codes.

`synthetic.py` already provides the A/B/C/scatter fixtures it will be tested against. The scenario's `ground_truth_label` is used for scoring only.
