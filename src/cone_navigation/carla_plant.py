"""CARLA plant: the controller is the only command source (no autopilot / TM)."""
from __future__ import annotations

import carla

from .actuation import ActuatorMapper
from .carla_session import CarlaSession
from .carla_world import read_ego_state


class CarlaPlant:
    name = "carla"

    def __init__(self, sess: CarlaSession, ego, mapper: ActuatorMapper, view: str | None = "chase"):
        self.sess, self.ego, self.mapper, self.view = sess, ego, mapper, view
        self._events: list[str] = []
        bp = sess.blueprint("sensor.other.collision")
        self.collision_sensor = sess.spawn(bp, carla.Transform(), attach_to=ego)
        self.collision_sensor.listen(lambda e: self._events.append(f"frame {e.frame}: {e.other_actor.type_id}"))
        self._last = None
        mapper.reset()
        self._update_view()

    def state(self):
        self._last = read_ego_state(self.sess, self.ego)
        return self._last

    def apply(self, accel: float, delta: float, v_next: float | None = None) -> dict:
        v_now = self._last.speed_mps if self._last else 0.0
        cmd = self.mapper.to_command(accel, delta, v_now, v_next)
        self.ego.apply_control(carla.VehicleControl(throttle=cmd.throttle, steer=cmd.steer, brake=cmd.brake,
                                                    hand_brake=False, reverse=False))
        return {"throttle": cmd.throttle, "steer": cmd.steer, "brake": cmd.brake}

    def step(self):
        self.sess.tick()
        self._update_view()

    def _update_view(self):
        if self.view == "chase":
            self.sess.follow_spectator(self.ego)

    def collision_events(self) -> list:
        return self._events
