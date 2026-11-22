"""
agents.py
=========
The signal agents of SmartFlow AI.

There is ONE independent SignalAgent per direction (North, South, East, West).
Each agent follows a simple perceive -> decide -> request cycle:

    1. PERCEIVE : look at its own lane (how many vehicles wait? how long has
                  the longest-waiting vehicle been waiting?)
    2. DECIDE   : turn that into an urgency score and a wanted green time
    3. REQUEST  : hand a PhaseRequest to whoever asks (the coordinator, later)

What an agent does NOT do
-------------------------
* It never touches the junction or any traffic signal. It does not even
  import the Junction class.
* It never looks at another agent's lane.
* It does not decide which phase actually gets green. It only says what it
  would like. The central coordinator (next module) decides what is allowed.

The two formulas (both easy to explain in a viva)
-------------------------------------------------
    urgency        = queue_weight * queue_length + wait_weight * max_wait

    requested_green = ceil(queue_length * seconds_per_vehicle)
                      then clamped into [min_green, max_green]
    where seconds_per_vehicle = 1 / discharge_rate
    (discharge_rate 0.5 vehicles/second -> 2 seconds per vehicle)

The wait term matters: a short queue whose first driver has waited a very long
time still gains urgency, so no direction is ignored forever.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

from config import Direction, SimulationConfig
from junction import Phase
from traffic import Lane

# Default weights. With these, 1 waiting vehicle counts as much as 2 seconds
# of waiting for the first vehicle. Both are configurable per agent.
DEFAULT_QUEUE_WEIGHT: float = 1.0
DEFAULT_WAIT_WEIGHT: float = 0.5


# --------------------------------------------------------------------------- #
# Small validation helpers (private to this file)
# --------------------------------------------------------------------------- #
def _require_count(name: str, value: Any) -> int:
    """Return ``value`` as an int, or raise ValueError unless it is a whole number >= 0."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a whole number, got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must not be negative, got {value!r}")
    return int(value)


def _require_weight(name: str, value: Any) -> float:
    """Return ``value`` as a float, or raise ValueError unless it is a finite number >= 0."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a number, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must not be negative, got {value!r}")
    return float(value)


# --------------------------------------------------------------------------- #
# The request an agent sends
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PhaseRequest:
    """What one agent asks for. Frozen, so nobody can change it after it is made.

    Attributes
    ----------
    direction:       the agent's direction (e.g. NORTH).
    phase:           the phase that direction belongs to (e.g. NORTH_SOUTH).
    urgency:         how badly the agent wants green (bigger = more urgent).
    requested_green: how many seconds of green it would like.
    queue_length:    vehicles waiting in its lane when the request was made.
    max_wait:        seconds the longest-waiting vehicle has waited (0 if none).
    """

    direction: Direction
    phase: Phase
    urgency: float
    requested_green: int
    queue_length: int
    max_wait: int

    @property
    def has_demand(self) -> bool:
        """True if at least one vehicle is waiting (an empty lane has no real demand)."""
        return self.queue_length > 0


# --------------------------------------------------------------------------- #
# The agent
# --------------------------------------------------------------------------- #
class SignalAgent:
    """An independent agent that represents ONE direction of the junction.

    Parameters
    ----------
    direction:     the direction this agent represents.
    lane:          the lane of that same direction (the only thing the agent observes).
    config:        simulation settings (min/max green and discharge rate are used).
    queue_weight:  importance of queue length in the urgency score (>= 0).
    wait_weight:   importance of the longest wait in the urgency score (>= 0).
                   At least one of the two weights must be above zero.

    The agent keeps no memory between calls: every request is based purely on
    what the lane looks like right now.
    """

    def __init__(
        self,
        direction: Direction,
        lane: Lane,
        config: SimulationConfig,
        queue_weight: float = DEFAULT_QUEUE_WEIGHT,
        wait_weight: float = DEFAULT_WAIT_WEIGHT,
    ) -> None:
        if not isinstance(direction, Direction):
            raise ValueError(f"direction must be a Direction, got {direction!r}")
        if not isinstance(lane, Lane):
            raise ValueError(f"lane must be a Lane, got {lane!r}")
        if not isinstance(config, SimulationConfig):
            raise ValueError(f"config must be a SimulationConfig, got {config!r}")
        if lane.direction is not direction:
            raise ValueError(
                f"an agent may only observe its own lane: agent is {direction.name}, "
                f"lane is {lane.direction.name}"
            )

        queue_weight = _require_weight("queue_weight", queue_weight)
        wait_weight = _require_weight("wait_weight", wait_weight)
        if queue_weight == 0 and wait_weight == 0:
            raise ValueError("queue_weight and wait_weight cannot both be zero")

        self.direction: Direction = direction
        self.lane: Lane = lane
        self.config: SimulationConfig = config
        self.phase: Phase = Phase.for_direction(direction)   # NORTH/SOUTH -> NORTH_SOUTH, ...
        self.queue_weight: float = queue_weight
        self.wait_weight: float = wait_weight

    # ----- 1. PERCEIVE -------------------------------------------------------
    def observe_queue_length(self) -> int:
        """How many vehicles are waiting in this agent's lane right now."""
        return self.lane.queue_length

    def observe_max_wait(self, current_time: int) -> int:
        """Seconds the longest-waiting vehicle has waited (0 for an empty lane).

        The lane is a FIFO queue, so the vehicle at the front is always the one
        that has waited longest.
        """
        current_time = _require_count("current_time", current_time)
        return self.lane.first_vehicle_waiting_time(current_time)

    # ----- 2. DECIDE ---------------------------------------------------------
    def calculate_urgency(self, queue_length: int, max_wait: int) -> float:
        """urgency = queue_weight * queue_length + wait_weight * max_wait"""
        queue_length = _require_count("queue_length", queue_length)
        max_wait = _require_count("max_wait", max_wait)
        return self.queue_weight * queue_length + self.wait_weight * max_wait

    def calculate_green_duration(self, queue_length: int) -> int:
        """Green seconds needed to clear the queue, kept within [min_green, max_green].

        Example with the default config (0.5 vehicles/second -> 2 s per vehicle):
            queue 3  -> 6 s   -> raised to min_green (10)
            queue 7  -> 14 s
            queue 30 -> 60 s  -> cut down to max_green (40)
            queue 0  -> 0 s   -> min_green (10)
        """
        queue_length = _require_count("queue_length", queue_length)
        seconds_per_vehicle = 1.0 / self.config.discharge_rate
        # round() first so float noise (e.g. 10.000000000000002) cannot add a spare second.
        seconds_needed = math.ceil(round(queue_length * seconds_per_vehicle, 6))
        return max(self.config.min_green, min(self.config.max_green, seconds_needed))

    # ----- 3. REQUEST --------------------------------------------------------
    def make_request(self, current_time: int) -> PhaseRequest:
        """Observe the lane, decide, and return a PhaseRequest.

        Does not change the lane and does not touch any signal.
        An empty lane still produces a request, with urgency 0.0, the minimum
        green time and ``has_demand == False``, so the coordinator can simply
        ignore it.
        """
        queue_length = self.observe_queue_length()
        max_wait = self.observe_max_wait(current_time)   # also validates current_time
        return PhaseRequest(
            direction=self.direction,
            phase=self.phase,
            urgency=self.calculate_urgency(queue_length, max_wait),
            requested_green=self.calculate_green_duration(queue_length),
            queue_length=queue_length,
            max_wait=max_wait,
        )
