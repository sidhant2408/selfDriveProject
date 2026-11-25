"""Closed-loop episode: observe -> plan (5 Hz) -> MPC (10 Hz) -> actuate -> tick (20 Hz).

Works with any plant exposing state() / apply(a, delta, v_next) / step() /
collision_events(): the offline KinematicPlant or the CARLA plant.

Per tick k (single owner, deterministic order):
    ego_k  = plant.state()                 # state of the frame just computed
    check termination on ego_k
    if k % plan_every == 0:  plan = planner.plan(ego_k, provider.get_cones(ego_k), ...)
    if k % ctrl_every == 0:  u = mpc.solve(ego_k, reference(plan))   (fallback brake on failure)
    plant.apply(u)                          # u held (zero-order hold) between MPC updates
    plant.step()                            # -> frame k+1
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .metrics import EpisodeMetrics
from .mpc import KinematicMPC, build_reference
from .planner import LocalPlanner, Plan


@dataclass
class EpisodeResult:
    outcome: str                 # GOAL | STOPPED | COLLISION | TIMEOUT
    summary: dict
    rows: list
    plans: list = field(default_factory=list)   # (t, Plan) snapshots, ~1 Hz


def run_episode(plant, provider, planner: LocalPlanner, mpc: KinematicMPC, road, scenario, geom,
                truth_cones, tick_dt: float, max_seconds: float, plan_every: int = 4,
                stop_stable_s: float = 2.0, on_tick=None) -> EpisodeResult:
    ctrl_every = max(1, round(mpc.cfg.dt / tick_dt))
    metrics = EpisodeMetrics(truth_cones, road, scenario.corridors, geom)
    plan: Plan | None = None
    res = None
    plans = []
    still = 0
    outcome = "TIMEOUT"
    n_ticks = int(round(max_seconds / tick_dt))
    t_start = None
    for k in range(n_ticks):
        ego = plant.state()
        t_start = ego.timestamp_s if t_start is None else t_start
        t = ego.timestamp_s - t_start
        s, d = road.reference.to_frenet(ego.x, ego.y)

        plan_time = None
        if k % plan_every == 0 or plan is None:
            cones = provider.get_cones(ego)
            t0 = time.perf_counter()
            plan = planner.plan(ego, cones, road, scenario.corridors)
            plan_time = time.perf_counter() - t0
            if k % (plan_every * 5) == 0:
                plans.append((t, plan))
        fresh = k % ctrl_every == 0 or res is None
        if fresh:
            res = mpc.solve(ego, build_reference(plan, ego, mpc.cfg))
            if not res.ok:
                metrics.solver_failures += 1
        cmd = plant.apply(res.accel, res.delta, res.v_next)
        row = metrics.update(t, ego, plan, res, cmd, plan_time, fresh)
        if on_tick:
            on_tick(k, ego, plan, res, row)

        # ---- termination (evaluated on the current, true state)
        metrics.collision_events = list(plant.collision_events())
        if row["clearance_m"] < 0 or metrics.collision_events:
            outcome = "COLLISION"
            break
        if scenario.goal.contains(s, d):
            outcome = "GOAL"
            break
        if plan.decision == "STOP" and abs(ego.speed_mps) < 0.05:
            still += 1
            if still * tick_dt >= stop_stable_s:
                outcome = "STOPPED"
                break
        else:
            still = 0
        plant.step()
    plans.append((metrics.rows[-1]["t_s"] if metrics.rows else 0.0, plan))
    summary = metrics.summary(outcome, tick_dt)
    return EpisodeResult(outcome, summary, metrics.rows, plans)


def expected_outcome(label: str | None) -> str:
    """EVALUATION ONLY: what a correct system should do for a scenario label."""
    return "STOPPED" if label == "CLOSURE" else "GOAL"
