"""
traffic.py
==========
The "traffic world" of SmartFlow AI: vehicles, lanes (queues) and random arrivals.

Three building blocks live here:

    Vehicle           - one car: ID, arrival time, departure time.
    Lane              - a FIFO queue of vehicles waiting at one approach.
    TrafficGenerator  - creates random vehicle arrivals for all four directions.

Nothing in this file knows about traffic signals, agents or controllers.
It only answers: "who arrived, who is waiting, who has left?"
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from config import Direction, SimulationConfig


def _require_time(name: str, value: Any) -> None:
    """Raise ValueError unless ``value`` is a whole number of seconds >= 0."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a whole number of seconds, got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must not be negative, got {value!r}")


# --------------------------------------------------------------------------- #
# Vehicle
# --------------------------------------------------------------------------- #
@dataclass
class Vehicle:
    """A single vehicle travelling through the junction.

    Attributes
    ----------
    vehicle_id:
        Unique number for this vehicle.
    arrival_time:
        Simulation second at which the vehicle joined the queue.
    departure_time:
        Simulation second at which it crossed the stop line.
        ``None`` means it is still waiting.
    """

    vehicle_id: int
    arrival_time: int
    departure_time: Optional[int] = None

    def __post_init__(self) -> None:
        _require_time("arrival_time", self.arrival_time)
        if self.departure_time is not None:
            _require_time("departure_time", self.departure_time)
            if self.departure_time < self.arrival_time:
                raise ValueError("departure_time cannot be earlier than arrival_time")

    @property
    def has_departed(self) -> bool:
        """True once the vehicle has crossed the stop line."""
        return self.departure_time is not None

    def mark_departed(self, time: int) -> None:
        """Record the moment the vehicle leaves the queue."""
        _require_time("departure time", time)
        if self.has_departed:
            raise ValueError(f"vehicle {self.vehicle_id} has already departed")
        if time < self.arrival_time:
            raise ValueError("a vehicle cannot depart before it arrives")
        self.departure_time = int(time)

    def waiting_time(self, current_time: int) -> int:
        """How many seconds this vehicle has waited.

        * Still in the queue:  ``current_time - arrival_time`` (it keeps growing).
        * Already departed:    ``departure_time - arrival_time`` (fixed forever).
        """
        _require_time("current_time", current_time)
        end_time = self.departure_time if self.has_departed else current_time
        if end_time < self.arrival_time:
            raise ValueError("current_time is earlier than the vehicle's arrival_time")
        return int(end_time - self.arrival_time)


# --------------------------------------------------------------------------- #
# Lane
# --------------------------------------------------------------------------- #
class Lane:
    """The queue of vehicles waiting on ONE approach (e.g. North).

    It is a FIFO queue: First In, First Out. The vehicle that has waited
    longest is always the next one to leave.
    """

    def __init__(self, direction: Direction) -> None:
        self.direction: Direction = direction
        self._queue: Deque[Vehicle] = deque()   # vehicles still waiting
        self._departed: List[Vehicle] = []      # vehicles that have left (kept for metrics)
        self._total_arrived: int = 0
        self._last_arrival_time: int = 0        # used to keep FIFO order honest

    # ----- changing the queue ------------------------------------------------
    def add_vehicle(self, vehicle: Vehicle) -> None:
        """Put a newly arrived vehicle at the BACK of the queue."""
        if vehicle.has_departed:
            raise ValueError("cannot add a vehicle that has already departed")
        if vehicle.arrival_time < self._last_arrival_time:
            raise ValueError(
                "vehicles must be added in order of arrival time "
                f"(got {vehicle.arrival_time} after {self._last_arrival_time})"
            )
        self._queue.append(vehicle)
        self._last_arrival_time = vehicle.arrival_time
        self._total_arrived += 1

    def remove_vehicle(self, current_time: int) -> Optional[Vehicle]:
        """Let the vehicle at the FRONT of the queue leave.

        Its ``departure_time`` is set to ``current_time``. Returns the vehicle,
        or ``None`` if the lane was empty (so callers can simply check the result).
        """
        if not self._queue:
            return None
        vehicle = self._queue[0]
        vehicle.mark_departed(current_time)  # validates first; queue is untouched if it fails
        self._queue.popleft()
        self._departed.append(vehicle)
        return vehicle

    # ----- looking at the queue (does not change anything) -------------------
    @property
    def queue_length(self) -> int:
        """Number of vehicles currently waiting."""
        return len(self._queue)

    def is_empty(self) -> bool:
        """True if nobody is waiting."""
        return not self._queue

    def peek(self) -> Optional[Vehicle]:
        """Look at the first vehicle without removing it (``None`` if empty)."""
        return self._queue[0] if self._queue else None

    def first_vehicle_waiting_time(self, current_time: int) -> int:
        """Seconds the vehicle at the front has waited so far (0 if the lane is empty).

        This is the "how long has the longest-waiting driver been stuck?" value
        that an agent can later use to avoid starving a lane.
        """
        if not self._queue:
            return 0
        return self._queue[0].waiting_time(current_time)

    def waiting_vehicles(self) -> Tuple[Vehicle, ...]:
        """Read-only snapshot of the vehicles still in the queue (front first)."""
        return tuple(self._queue)

    @property
    def departed_vehicles(self) -> Tuple[Vehicle, ...]:
        """Read-only snapshot of every vehicle that has left this lane."""
        return tuple(self._departed)

    @property
    def total_arrived(self) -> int:
        """How many vehicles have ever joined this lane."""
        return self._total_arrived

    @property
    def total_departed(self) -> int:
        """How many vehicles have left this lane (this is the lane's throughput)."""
        return len(self._departed)


# --------------------------------------------------------------------------- #
# TrafficGenerator
# --------------------------------------------------------------------------- #
class TrafficGenerator:
    """Creates random vehicle arrivals for North, South, East and West.

    How it works (in simple words)
    ------------------------------
    * Each second, the number of new vehicles in a direction is drawn from a
      Poisson distribution whose average is that direction's arrival rate.
      (Poisson is the standard model for random, independent arrivals.)
    * Every direction has its OWN random-number stream, derived from the one
      master seed. So the four directions are independent of each other, and
      changing the East rate does not change the North traffic at all.
    * The whole arrival schedule is drawn once, when the generator is created.
      Two generators built from the same config therefore produce exactly the
      same traffic. This is what lets us give the fixed-time and the adaptive
      strategy the very same vehicles.

    Usage: create ONE generator per simulation run, then call ``generate(t)``
    for t = 0, 1, 2, ... in order.
    """

    def __init__(self, config: SimulationConfig) -> None:
        self.config: SimulationConfig = config

        # One independent child seed per direction, all derived from the master seed.
        child_seeds = np.random.SeedSequence(config.seed).spawn(len(Direction))

        # Pre-draw "how many vehicles arrive in each second" for every direction.
        self._arrival_counts: Dict[Direction, np.ndarray] = {}
        for direction, child_seed in zip(Direction, child_seeds):
            rng = np.random.default_rng(child_seed)
            self._arrival_counts[direction] = rng.poisson(
                lam=config.arrival_rate(direction), size=config.duration
            )

        self._next_vehicle_id: int = 1

    def generate(self, time_step: int) -> Dict[Direction, List[Vehicle]]:
        """Create the vehicles that arrive during second ``time_step``.

        Returns a dictionary such as ``{Direction.NORTH: [Vehicle, ...], ...}``.
        Every direction is always present (the list is empty if nobody arrived).
        Vehicle IDs are unique across all directions.
        """
        _require_time("time_step", time_step)
        if time_step >= self.config.duration:
            raise ValueError(
                f"time_step {time_step} is outside the simulation (duration={self.config.duration})"
            )

        new_vehicles: Dict[Direction, List[Vehicle]] = {}
        for direction in Direction:
            count = int(self._arrival_counts[direction][time_step])
            vehicles: List[Vehicle] = []
            for _ in range(count):
                vehicles.append(Vehicle(vehicle_id=self._next_vehicle_id, arrival_time=int(time_step)))
                self._next_vehicle_id += 1
            new_vehicles[direction] = vehicles
        return new_vehicles

    def arrival_counts(self, direction: Direction) -> np.ndarray:
        """Copy of the pre-drawn arrivals-per-second for one direction (length = duration)."""
        return self._arrival_counts[direction].copy()

    def total_arrivals(self, direction: Direction) -> int:
        """Total number of vehicles that will arrive in this direction over the whole run."""
        return int(self._arrival_counts[direction].sum())

    def reset(self) -> None:
        """Restart vehicle numbering at 1 so the same run can be replayed.

        The arrival schedule itself never changes, only the ID counter restarts.
        """
        self._next_vehicle_id = 1
