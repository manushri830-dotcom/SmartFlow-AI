"""
controllers.py
==============
The two traffic controllers of SmartFlow AI, behind ONE common interface.

    TrafficController      - the common interface (an abstract base class).
    FixedTimeController    - baseline: alternates the two phases on a timer.
    AdaptiveController     - multi-agent: agents + coordinator react to traffic.

Why a common interface?
-----------------------
The simulator (next modules) must be able to run EITHER controller with the
very same loop. So both controllers offer the same method:

    decision = controller.update(current_time)

The simulator's loop for one second therefore looks like this, and it never
needs to know which controller it is driving:

    1. add this second's vehicle arrivals to the lanes
    2. decision = controller.update(current_time)
    3. discharge vehicles on the green directions
    4. controller.junction.advance()

Who owns what?
--------------
* The SIMULATOR creates the Junction and the Lanes (fresh ones for every run,
  so each controller starts from identical initial conditions) and feeds the
  same pre-generated arrivals to both runs. That is what keeps the comparison fair.
* A controller only DECIDES. It never creates traffic, never removes vehicles
  and never calls ``junction.advance()``. It talks to the Junction through the
  Junction's public methods (start_phase) only, so the safe sequence
  GREEN -> YELLOW -> ALL_RED -> GREEN is always enforced by the Junction itself.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Dict, Optional

from agents import DEFAULT_QUEUE_WEIGHT, DEFAULT_WAIT_WEIGHT, SignalAgent
from config import Direction
from coordinator import Action, CoordinatorDecision, TrafficCoordinator
from junction import Junction, Phase
from traffic import Lane


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _require_time(name: str, value: Any) -> int:
    """Return ``value`` as an int, or raise ValueError unless it is a whole number >= 0."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a whole number of seconds, got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must not be negative, got {value!r}")
    return int(value)


# --------------------------------------------------------------------------- #
# The result every controller returns
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ControllerDecision:
    """What a controller did to the junction during one call to ``update()``.

    It reuses the coordinator's ``Action`` enum (NONE, START, QUEUE, EXTEND) so
    both controllers speak the same language.

    Attributes
    ----------
    action:         what was sent to the junction this second.
    phase:          the phase involved (started / queued / extended), or the
                    phase that is coming up next when nothing was sent. May be
                    ``None`` when the junction is idle and nobody wants green.
    green_duration: seconds of green started or added (0 when nothing was sent).
    """

    action: Action
    phase: Optional[Phase]
    green_duration: int


# --------------------------------------------------------------------------- #
# The common interface
# --------------------------------------------------------------------------- #
class TrafficController(ABC):
    """Base class of every controller: it owns (a reference to) one Junction.

    Subclasses must implement ``update(current_time)``.
    """

    #: Human-readable name, handy for plots and result tables later.
    name: str = "Controller"

    def __init__(self, junction: Junction) -> None:
        if not isinstance(junction, Junction):
            raise ValueError(f"junction must be a Junction, got {junction!r}")
        self._junction: Junction = junction

    @property
    def junction(self) -> Junction:
        """The junction this controller drives (the simulator calls ``advance()`` on it)."""
        return self._junction

    @abstractmethod
    def update(self, current_time: int) -> ControllerDecision:
        """Decide what to do this second and instruct the junction if needed.

        Call exactly once per simulation second, BEFORE ``junction.advance()``.
        """


# --------------------------------------------------------------------------- #
# Baseline: fixed-time controller
# --------------------------------------------------------------------------- #
class FixedTimeController(TrafficController):
    """Alternates NORTH_SOUTH and EAST_WEST, each for ``config.fixed_green`` seconds.

    It is deliberately "blind": it never looks at a lane, so it cannot react to
    traffic. With the default config the cycle repeats every 24 seconds:

        20 s green (one phase) + 3 s yellow + 1 s all-red, then the other phase.

    How it works with the Junction
    ------------------------------
    * Junction IDLE (only at the very start): start NORTH_SOUTH.
    * Otherwise, as soon as no next phase is queued, queue the OTHER phase.
      The Junction keeps the current green running to its end (it is never cut
      short), then runs yellow and all-red, and only then starts the queued phase.
    * Because the next phase is always queued, the junction never goes back to
      IDLE, so the controller keeps cycling even when there is no traffic at all.
      That is exactly what a real fixed-time signal does.
    """

    name = "Fixed-Time"

    def __init__(self, junction: Junction) -> None:
        super().__init__(junction)
        self._green_duration: int = junction.config.fixed_green

    @property
    def green_duration(self) -> int:
        """The green time (seconds) given to every phase."""
        return self._green_duration

    def update(self, current_time: int) -> ControllerDecision:
        _require_time("current_time", current_time)   # time is validated, but never used to decide
        junction = self._junction

        if junction.is_idle:
            junction.start_phase(Phase.NORTH_SOUTH, self._green_duration)
            return ControllerDecision(Action.START, Phase.NORTH_SOUTH, self._green_duration)

        if junction.pending_phase is None:
            next_phase = junction.active_phase.other
            junction.start_phase(next_phase, self._green_duration)   # queued by the junction
            return ControllerDecision(Action.QUEUE, next_phase, self._green_duration)

        return ControllerDecision(Action.NONE, junction.pending_phase, 0)


# --------------------------------------------------------------------------- #
# Multi-agent: adaptive controller
# --------------------------------------------------------------------------- #
class AdaptiveController(TrafficController):
    """Adaptive controller built from the existing SignalAgents and TrafficCoordinator.

    Parameters
    ----------
    junction:
        The Junction to drive.
    lanes:
        ``{Direction: Lane}`` with exactly one lane per direction. These must be
        the same Lane objects the simulator fills and empties. One SignalAgent
        is created for each lane.
    starvation_threshold:
        Passed to the coordinator. ``None`` uses the coordinator's default.
    queue_weight, wait_weight:
        Urgency weights given to every agent.

    This class adds almost no logic of its own: the agents measure urgency,
    the coordinator decides, the junction keeps everything safe. The controller
    only wires them together and presents them through the common interface.
    """

    name = "Adaptive Multi-Agent"

    def __init__(
        self,
        junction: Junction,
        lanes: Mapping,
        starvation_threshold: Optional[int] = None,
        queue_weight: float = DEFAULT_QUEUE_WEIGHT,
        wait_weight: float = DEFAULT_WAIT_WEIGHT,
    ) -> None:
        super().__init__(junction)
        self._lanes: Dict[Direction, Lane] = self._check_lanes(lanes)

        # SignalAgent checks that each lane really belongs to its direction.
        agents = [
            SignalAgent(
                direction,
                self._lanes[direction],
                junction.config,
                queue_weight=queue_weight,
                wait_weight=wait_weight,
            )
            for direction in Direction
        ]
        self._coordinator = TrafficCoordinator(junction, agents, starvation_threshold)
        self._last_decision: Optional[CoordinatorDecision] = None

    @staticmethod
    def _check_lanes(lanes: Any) -> Dict[Direction, Lane]:
        """Check that there is exactly one Lane for each of the four directions."""
        if not isinstance(lanes, Mapping):
            raise ValueError(f"lanes must be a mapping {{Direction: Lane}}, got {lanes!r}")
        if set(lanes) != set(Direction):
            raise ValueError("lanes must contain exactly one lane for each of the four directions")
        for direction, lane in lanes.items():
            if not isinstance(lane, Lane):
                raise ValueError(f"lane for {direction.name} must be a Lane, got {lane!r}")
        return {direction: lanes[direction] for direction in Direction}   # fixed N, S, E, W order

    # ----- read-only information ---------------------------------------------
    @property
    def coordinator(self) -> TrafficCoordinator:
        """The coordinator used by this controller."""
        return self._coordinator

    @property
    def lanes(self) -> Dict[Direction, Lane]:
        """The four lanes as ``{Direction: Lane}`` (a copy of the dictionary)."""
        return dict(self._lanes)

    @property
    def last_coordinator_decision(self) -> Optional[CoordinatorDecision]:
        """The full coordinator decision (including the Reason) from the latest update."""
        return self._last_decision

    # ----- the common interface ----------------------------------------------
    def update(self, current_time: int) -> ControllerDecision:
        decision = self._coordinator.update(current_time)   # validates current_time too
        self._last_decision = decision
        return ControllerDecision(decision.action, decision.phase, decision.green_duration)
