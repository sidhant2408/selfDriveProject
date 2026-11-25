"""Exact cone data provider (stand-in for perception).

This is the ONLY route by which pipeline modules receive cone data in
ground-truth mode. Phase 5's noise adapter will wrap a provider with the same
``get_cones`` signature; Phase 6 perception will implement it too.
"""
from __future__ import annotations

import copy

from .observations import ConeObservation
from .state import EgoState


class GroundTruthConeProvider:
    def __init__(self, truth: list[ConeObservation]):
        self._truth = copy.deepcopy(truth)

    def get_cones(self, ego_state: EgoState | None = None) -> list[ConeObservation]:
        # Ground truth ignores ego_state (no field-of-view limit). Return copies so
        # a consumer can never mutate the truth used later for evaluation.
        return copy.deepcopy(self._truth)
