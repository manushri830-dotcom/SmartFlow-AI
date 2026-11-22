"""
config.py
=========
Central settings for SmartFlow AI (Multi-Agent Adaptive Traffic Signal Simulation).

Every other module reads its settings from a single ``SimulationConfig`` object,
so there is exactly ONE place to look when you want to know "what are the
current parameters of this simulation?".

Time convention used everywhere in the project:
    * One simulation step = 1 second.
    * All durations (simulation length, green, yellow, all-red) are whole seconds.
    * Arrival rates are "vehicles per second" for ONE direction.
      (0.15 vehicles/second is about 540 vehicles/hour.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from numbers import Integral, Real
from typing import Any, Dict


class Direction(Enum):
    """The four approaches (arms) of the junction.

    Iterating over ``Direction`` always gives the same order:
    North, South, East, West. A fixed order matters for reproducibility.
    """

    NORTH = "North"
    SOUTH = "South"
    EAST = "East"
    WEST = "West"


# --------------------------------------------------------------------------- #
# Small validation helpers (kept private to this file)
# --------------------------------------------------------------------------- #
def _require_int(name: str, value: Any, minimum: int) -> None:
    """Raise ValueError unless ``value`` is a whole number >= ``minimum``."""
    # bool is a subclass of int in Python (True == 1), so reject it explicitly.
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a whole number, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value!r}")


def _require_number(name: str, value: Any, minimum: float, strictly_above: bool = False) -> None:
    """Raise ValueError unless ``value`` is a finite real number above ``minimum``.

    If ``strictly_above`` is True the value must be > minimum, otherwise >= minimum.
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a number, got {value!r}")
    if not math.isfinite(value):  # rejects nan and +/- infinity
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    if strictly_above and value <= minimum:
        raise ValueError(f"{name} must be greater than {minimum}, got {value!r}")
    if not strictly_above and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value!r}")


# --------------------------------------------------------------------------- #
# The configuration object
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SimulationConfig:
    """All tunable parameters of one simulation run.

    The object is *frozen*: once created (and validated) it cannot be changed
    by accident. To try a different value, build a modified copy:

        >>> from dataclasses import replace
        >>> busy = replace(SimulationConfig(), arrival_rate_north=0.4)

    ``replace`` runs the validation again, so a bad copy is rejected too.

    Attributes
    ----------
    duration:
        Length of the simulation in seconds (= number of time steps).
    seed:
        Random seed. The same seed + same settings always gives the same traffic.
    arrival_rate_north/south/east/west:
        Average vehicles arriving per second in that direction.
    min_green / max_green:
        Smallest and largest green time (seconds) a signal may be given.
    yellow_duration:
        Seconds of yellow shown between a green and the next phase.
    all_red_duration:
        Seconds when every signal is red, to clear the junction safely.
    discharge_rate:
        Vehicles that can leave a lane per second of green (0.5 = one vehicle
        every 2 seconds).
    fixed_green:
        Green time (seconds) used by the traditional fixed-time strategy.
    """

    # --- simulation control ---
    duration: int = 3600
    seed: int = 42

    # --- traffic demand (vehicles per second, per direction) ---
    arrival_rate_north: float = 0.15
    arrival_rate_south: float = 0.15
    arrival_rate_east: float = 0.15
    arrival_rate_west: float = 0.15

    # --- signal timing (seconds) ---
    min_green: int = 10
    max_green: int = 40
    yellow_duration: int = 3
    all_red_duration: int = 1
    fixed_green: int = 20

    # --- junction capacity ---
    discharge_rate: float = 0.5

    def __post_init__(self) -> None:
        """Validate every field. Runs automatically right after creation."""
        # 1) Check each field on its own (type and range).
        _require_int("duration", self.duration, minimum=1)
        _require_int("seed", self.seed, minimum=0)

        for name in (
            "arrival_rate_north",
            "arrival_rate_south",
            "arrival_rate_east",
            "arrival_rate_west",
        ):
            _require_number(name, getattr(self, name), minimum=0.0)

        _require_int("min_green", self.min_green, minimum=1)
        _require_int("max_green", self.max_green, minimum=1)
        _require_int("yellow_duration", self.yellow_duration, minimum=1)  # yellow is a safety step
        _require_int("all_red_duration", self.all_red_duration, minimum=0)
        _require_int("fixed_green", self.fixed_green, minimum=1)
        _require_number("discharge_rate", self.discharge_rate, minimum=0.0, strictly_above=True)

        # 2) Check relationships between fields.
        if self.max_green < self.min_green:
            raise ValueError(
                f"max_green ({self.max_green}) must not be smaller than min_green ({self.min_green})"
            )
        # The fixed-time baseline stays inside the same green limits as the
        # adaptive strategy, so the comparison between them is fair.
        if not (self.min_green <= self.fixed_green <= self.max_green):
            raise ValueError(
                f"fixed_green ({self.fixed_green}) must be between "
                f"min_green ({self.min_green}) and max_green ({self.max_green})"
            )

    @property
    def arrival_rates(self) -> Dict[Direction, float]:
        """Arrival rate of every direction as a dictionary: {Direction: rate}."""
        return {
            Direction.NORTH: self.arrival_rate_north,
            Direction.SOUTH: self.arrival_rate_south,
            Direction.EAST: self.arrival_rate_east,
            Direction.WEST: self.arrival_rate_west,
        }

    def arrival_rate(self, direction: Direction) -> float:
        """Arrival rate (vehicles/second) of one direction."""
        return self.arrival_rates[direction]
