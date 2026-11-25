"""CARLA connection, synchronous stepping, actor ownership and cleanup.

Usage::

    with CarlaSession(cfg["simulation"], map_name="Town10HD_Opt") as sess:
        actor = sess.spawn(bp, transform)
        frame = sess.tick()
    # actors destroyed, original world settings restored, even on exceptions.

Written against the CARLA 1.0 (UE 5.8) Python API bundled with the package.
"""
from __future__ import annotations

import time

import carla


class CarlaSetupError(RuntimeError):
    pass


def _map_basename(name: str) -> str:
    return name.replace("\\", "/").rstrip("/").split("/")[-1].lower()


class CarlaSession:
    def __init__(self, sim_cfg: dict, map_name: str | None = None, realtime: bool = False):
        # realtime=True paces tick() to wall-clock (one fixed_dt per fixed_dt
        # seconds) so a human can watch. Simulation results are unchanged.
        self.realtime = realtime
        self._last_tick_wall: float | None = None
        self.host = sim_cfg.get("host", "127.0.0.1")
        self.port = int(sim_cfg.get("port", 2000))
        self.timeout_s = float(sim_cfg.get("timeout_s", 30.0))
        self.tick_timeout_s = float(sim_cfg.get("tick_timeout_s", 10.0))
        self.tick_timeouts: list[dict] = []
        self.fixed_dt = float(sim_cfg.get("fixed_delta_seconds", 0.05))
        self.no_rendering = bool(sim_cfg.get("no_rendering_mode", False))
        self.require_version_match = bool(sim_cfg.get("require_version_match", True))
        self.map_name = map_name
        self.client: carla.Client | None = None
        self.world: carla.World | None = None
        self.map: carla.Map | None = None
        self.versions: dict = {}
        self._original_settings = None
        self._actors: list = []

    # ------------------------------------------------------------------ lifecycle
    def __enter__(self) -> "CarlaSession":
        try:
            self._connect()
            self._load_map()
            self._enable_sync()
        except BaseException:
            self._restore()
            raise
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.destroy_actors()
        finally:
            self._restore()
        return False

    def _connect(self):
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(self.timeout_s)
        try:
            server_v = self.client.get_server_version()
        except RuntimeError as e:
            raise CarlaSetupError(
                f"Cannot reach a CARLA server at {self.host}:{self.port} ({e}). "
                "Start CarlaUnreal.exe first and check the RPC port (-carla-rpc-port)."
            ) from e
        client_v = self.client.get_client_version()
        self.versions = {"client": client_v, "server": server_v}
        if client_v != server_v:
            msg = (f"CARLA client {client_v} != server {server_v}. Install the client wheel "
                   "that ships with this server package.")
            if self.require_version_match:
                raise CarlaSetupError(msg)
            print(f"[carla] WARNING: {msg}")
        self.world = self.client.get_world()

    def _load_map(self):
        if self.map_name:
            current = _map_basename(self.world.get_map().name)
            wanted = _map_basename(self.map_name)
            if current != wanted:
                available = self.client.get_available_maps()
                if wanted not in {_map_basename(m) for m in available}:
                    raise CarlaSetupError(
                        f"Map '{self.map_name}' is not available on this server. "
                        f"Available: {available}")
                print(f"[carla] loading map {self.map_name} (was {current}) ...")
                self.world = self.client.load_world(self.map_name)
        self.map = self.world.get_map()

    def _enable_sync(self):
        self._original_settings = self.world.get_settings()
        s = self.world.get_settings()
        s.synchronous_mode = True
        s.fixed_delta_seconds = self.fixed_dt
        s.no_rendering_mode = self.no_rendering
        self.world.apply_settings(s)
        self.world.tick()
        got = self.world.get_settings()
        if not got.synchronous_mode or abs((got.fixed_delta_seconds or 0) - self.fixed_dt) > 1e-9:
            raise CarlaSetupError(f"World settings did not apply: sync={got.synchronous_mode} "
                                  f"dt={got.fixed_delta_seconds}")

    def _restore(self):
        if self.world is not None and self._original_settings is not None:
            try:
                self.world.apply_settings(self._original_settings)
            except RuntimeError as e:
                print(f"[carla] WARNING: could not restore world settings: {e}")
            self._original_settings = None

    # -------------------------------------------------------------------- actors
    def blueprint(self, bp_id: str):
        lib = self.world.get_blueprint_library()
        matches = [b for b in lib.filter(bp_id) if b.id == bp_id]
        if not matches:
            stem = bp_id.split(".")[-1][:4]
            similar = [b.id for b in lib if stem in b.id][:10]
            raise CarlaSetupError(f"Blueprint '{bp_id}' not found on this server. Similar: {similar}")
        return matches[0]

    def spawn(self, bp, transform, attach_to=None):
        """Spawn and track an actor; returns None if the location is blocked."""
        actor = self.world.try_spawn_actor(bp, transform, attach_to=attach_to) if attach_to \
            else self.world.try_spawn_actor(bp, transform)
        if actor is not None:
            self._actors.append(actor)
        return actor

    def destroy_actors(self):
        if not self._actors or self.client is None:
            return
        for a in self._actors:
            if isinstance(a, carla.Sensor) and a.is_listening:
                a.stop()
        cmds = [carla.command.DestroyActor(a) for a in reversed(self._actors)]
        self.client.apply_batch_sync(cmds, False)
        self._actors.clear()
        if self.world.get_settings().synchronous_mode:
            self.world.tick()

    # ---------------------------------------------------------------------- time
    def tick(self) -> int:
        """Advance one fixed step. On a timeout, check whether the server actually
        advanced (reply lost/late) before retrying once, so a single slow frame
        neither hangs the run nor silently double-steps it."""
        if self.realtime:
            now = time.perf_counter()
            if self._last_tick_wall is not None:
                time.sleep(max(0.0, self._last_tick_wall + self.fixed_dt - now))
            self._last_tick_wall = time.perf_counter()
        before = self.world.get_snapshot().frame
        try:
            frame = self.world.tick(self.tick_timeout_s)
        except RuntimeError as e:
            after = self.world.get_snapshot().frame
            self.tick_timeouts.append({"frame_before": before, "frame_after": after, "error": str(e)})
            print(f"[carla] WARNING: tick timed out after {self.tick_timeout_s}s "
                  f"(server frame {before} -> {after}); ", end="")
            if after > before:
                print("server did advance; continuing with that frame")
                return after
            print("retrying once")
            frame = self.world.tick(self.tick_timeout_s)
        return frame

    def follow_spectator(self, actor, back_m: float = 9.0, up_m: float = 4.5, pitch_deg: float = -18.0):
        """Chase camera behind `actor` (call after every tick)."""
        tf = actor.get_transform()
        fwd = tf.get_forward_vector()
        loc = tf.location - carla.Location(x=fwd.x * back_m, y=fwd.y * back_m) + carla.Location(z=up_m)
        self.world.get_spectator().set_transform(
            carla.Transform(loc, carla.Rotation(pitch=pitch_deg, yaw=tf.rotation.yaw)))

    def set_spectator_topdown(self, x_c: float, y_c: float, height: float = 45.0):
        self.world.get_spectator().set_transform(
            carla.Transform(carla.Location(x=x_c, y=y_c, z=height), carla.Rotation(pitch=-90.0)))
