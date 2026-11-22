"""
simulator.py
============
The simulation engine of SmartFlow AI. It connects everything built so far:

    TrafficSchedule  -> WHO arrives WHEN (drawn once, immutable, shared by both runs)
    Lane             -> the queues
    Junction         -> the safe signal state machine
    Controller       -> the decision maker (fixed-time OR adaptive)

The simulator OWNS time, arrivals, departures and the history. A controller
only decides; it never creates, removes or moves vehicles, and never advances
the clock (the simulator checks this every second).

Fair comparison
---------------
``Vehicle`` objects are mutable (they get a departure time), so two runs cannot
share them. Instead the random traffic is drawn ONCE by the existing
``TrafficGenerator`` and frozen into a ``TrafficSchedule`` (vehicle IDs per
second per direction). Every run builds its OWN fresh vehicles and lanes from
that schedule. Same schedule + same config = same arrivals, same initial
conditions, same discharge rule, same signal timing. Only the controller differs.

The loop for ONE simulation second ``t`` (this order is fixed)
--------------------------------------------------------------
    1. ARRIVE    vehicles scheduled for second t join the BACK of their lanes
                 (arrival_time = t).
    2. DECIDE    controller.update(t) looks at the lanes (including this
                 second's arrivals) and instructs the junction.
    3. DEPART    the signals now showing are the signals shown DURING second t.
                 Only GREEN directions discharge vehicles (departure_time = t).
    4. RECORD    one StepRecord: signals of second t, queues at the END of
                 second t, vehicles that arrived / departed in second t.
    5. ADVANCE   junction.advance() lets the second pass.

This matches the Junction's own rule: "read the signals, discharge vehicles,
THEN call advance()".

Conventions (so there are no off-by-one surprises)
--------------------------------------------------
* Time starts at 0 and the last simulated second is ``duration - 1``.
* A vehicle that arrives and leaves in the same second waits 0 seconds.
  Waiting time = departure_time - arrival_time. Nothing has to be "updated"
  every second: the waiting time is always derived from those two numbers.
  A vehicle still queued at the end is measured up to ``duration``.
* Discharge rate (vehicles per second) is applied with exact fractions: every
  green second a lane earns ``discharge_rate`` "credit"; each whole credit lets
  one vehicle leave. At 0.5 vehicles/second: second 1 of a green earns 0.5
  (nobody leaves), second 2 earns 1.0 (one vehicle leaves), and so on.
  Credit is lost when the direction is not green or its lane is empty.
  Because of that, a lane cannot "save up" capacity while idle.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from numbers import Integral
from typing import Any, Dict, List, Optional, Tuple

from config import Direction, SimulationConfig
from controllers import (
    AdaptiveController,
    ControllerDecision,
    FixedTimeController,
    TrafficController,
)
from junction import Junction, Phase, SignalState, Stage
from traffic import Lane, TrafficGenerator, Vehicle

_DIRECTIONS: Tuple[Direction, ...] = tuple(Direction)
_ADAPTIVE_OPTIONS: Tuple[str, ...] = ("starvation_threshold", "queue_weight", "wait_weight")


def _require_whole(name: str, value: Any, minimum: int = 0) -> int:
    """Return ``value`` as an int, or raise ValueError unless it is a whole number >= minimum."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a whole number, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value!r}")
    return int(value)


# --------------------------------------------------------------------------- #
# TrafficSchedule: the shared, immutable arrival plan
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TrafficSchedule:
    """Who arrives, when and where, for the WHOLE run. Frozen, so it can be shared safely.

    ``arrivals[t][i]`` is a tuple of the vehicle IDs that arrive during second
    ``t`` in direction ``i`` (directions in the fixed order of ``Direction``:
    North, South, East, West).

    Two schedules built from the same config compare equal, which is how the
    tests prove reproducibility.
    """

    config: SimulationConfig
    arrivals: Tuple[Tuple[Tuple[int, ...], ...], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.config, SimulationConfig):
            raise ValueError(f"config must be a SimulationConfig, got {self.config!r}")
        if not isinstance(self.arrivals, tuple):
            raise ValueError("arrivals must be a tuple (one entry per second)")
        if len(self.arrivals) != self.config.duration:
            raise ValueError(
                f"arrivals must have one entry per second: expected {self.config.duration}, "
                f"got {len(self.arrivals)}"
            )

        seen = set()
        for time_step, per_direction in enumerate(self.arrivals):
            if not isinstance(per_direction, tuple) or len(per_direction) != len(_DIRECTIONS):
                raise ValueError(f"second {time_step}: expected one tuple per direction")
            for ids in per_direction:
                if not isinstance(ids, tuple):
                    raise ValueError(f"second {time_step}: vehicle IDs must be given as a tuple")
                for vehicle_id in ids:
                    _require_whole("vehicle_id", vehicle_id, minimum=1)
                    if vehicle_id in seen:
                        raise ValueError(f"duplicate vehicle_id {vehicle_id}")
                    seen.add(vehicle_id)

    @classmethod
    def from_config(cls, config: SimulationConfig) -> "TrafficSchedule":
        """Draw the whole schedule ONCE with the existing, reproducible TrafficGenerator."""
        if not isinstance(config, SimulationConfig):
            raise ValueError(f"config must be a SimulationConfig, got {config!r}")
        generator = TrafficGenerator(config)
        arrivals = []
        for time_step in range(config.duration):
            new_vehicles = generator.generate(time_step)
            arrivals.append(
                tuple(tuple(vehicle.vehicle_id for vehicle in new_vehicles[d]) for d in _DIRECTIONS)
            )
        return cls(config, tuple(arrivals))

    @property
    def duration(self) -> int:
        """Number of simulated seconds covered by the schedule."""
        return len(self.arrivals)

    def vehicle_ids(self, time_step: int, direction: Direction) -> Tuple[int, ...]:
        """IDs of the vehicles that arrive in ``direction`` during second ``time_step``."""
        time_step = _require_whole("time_step", time_step)
        if time_step >= self.duration:
            raise ValueError(f"time_step {time_step} is outside the schedule (duration={self.duration})")
        if not isinstance(direction, Direction):
            raise ValueError(f"direction must be a Direction, got {direction!r}")
        return self.arrivals[time_step][_DIRECTIONS.index(direction)]

    def arrival_count(self, time_step: int, direction: Direction) -> int:
        """How many vehicles arrive in ``direction`` during second ``time_step``."""
        return len(self.vehicle_ids(time_step, direction))

    def total_arrivals(self, direction: Optional[Direction] = None) -> int:
        """Total vehicles in the whole schedule (one direction, or all four if ``None``)."""
        if direction is not None and not isinstance(direction, Direction):
            raise ValueError(f"direction must be a Direction or None, got {direction!r}")
        total = 0
        for per_direction in self.arrivals:
            for index, d in enumerate(_DIRECTIONS):
                if direction is None or direction is d:
                    total += len(per_direction[index])
        return total


# --------------------------------------------------------------------------- #
# Result types (plain, structured data; no plotting code here)
# --------------------------------------------------------------------------- #
class ControllerType(Enum):
    """Which controller a Simulator runs."""

    FIXED_TIME = "fixed_time"
    ADAPTIVE = "adaptive"


@dataclass(frozen=True)
class VehicleRecord:
    """A finished-or-waiting vehicle, frozen for later analysis (metrics module)."""

    vehicle_id: int
    direction: Direction
    arrival_time: int
    departure_time: Optional[int] = None   # None = still waiting when the run ended

    @property
    def has_departed(self) -> bool:
        return self.departure_time is not None

    def waiting_time(self, current_time: int) -> int:
        """Seconds waited. Departed: fixed forever. Still waiting: ``current_time - arrival_time``.

        For vehicles still queued at the end of a run, use ``current_time = duration``.
        """
        current_time = _require_whole("current_time", current_time)
        end_time = self.departure_time if self.has_departed else current_time
        if end_time < self.arrival_time:
            raise ValueError("current_time is earlier than the vehicle's arrival_time")
        return end_time - self.arrival_time


@dataclass(frozen=True)
class StepRecord:
    """Everything worth knowing about ONE simulated second (treat the dicts as read-only).

    Attributes
    ----------
    time:          the second (0 .. duration-1).
    stage:         junction stage shown during this second.
    active_phase:  the phase that is green/yellow/clearing, or None when idle.
    signals:       colour of each direction during this second.
    queue_lengths: vehicles waiting in each lane at the END of this second
                   (after this second's arrivals and departures).
    arrived:       vehicles that arrived in each direction during this second.
    departed:      vehicles that left each lane during this second.
    total_arrived / total_departed: cumulative counts up to and including this second.
    decision:      what the controller did to the junction this second.
    """

    time: int
    stage: Stage
    active_phase: Optional[Phase]
    signals: Dict[Direction, SignalState]
    queue_lengths: Dict[Direction, int]
    arrived: Dict[Direction, int]
    departed: Dict[Direction, int]
    total_arrived: int
    total_departed: int
    decision: ControllerDecision

    @property
    def total_waiting(self) -> int:
        """Vehicles waiting in all four lanes at the end of this second."""
        return sum(self.queue_lengths.values())


@dataclass(frozen=True)
class SimulationResult:
    """The outcome of one run (or of the part of it that has been simulated so far)."""

    config: SimulationConfig
    controller_type: ControllerType
    controller_name: str
    history: Tuple[StepRecord, ...]
    vehicles: Tuple[VehicleRecord, ...]   # every vehicle that arrived, ordered by arrival

    @property
    def duration(self) -> int:
        return self.config.duration

    @property
    def steps_completed(self) -> int:
        return len(self.history)

    @property
    def is_complete(self) -> bool:
        """True if the whole duration has been simulated."""
        return self.steps_completed == self.config.duration

    @property
    def total_arrived(self) -> int:
        return len(self.vehicles)

    @property
    def total_departed(self) -> int:
        return sum(1 for vehicle in self.vehicles if vehicle.has_departed)

    @property
    def total_waiting(self) -> int:
        return self.total_arrived - self.total_departed

    @property
    def final_queue_lengths(self) -> Dict[Direction, int]:
        """Queue length of each lane at the end of the last simulated second."""
        if not self.history:
            return {direction: 0 for direction in Direction}
        return dict(self.history[-1].queue_lengths)


@dataclass(frozen=True)
class ComparisonResult:
    """Both controllers, run on the very same schedule."""

    schedule: TrafficSchedule
    fixed_time: SimulationResult
    adaptive: SimulationResult


# --------------------------------------------------------------------------- #
# The simulator
# --------------------------------------------------------------------------- #
class Simulator:
    """Runs ONE controller on ONE traffic schedule.

    Parameters
    ----------
    config:            all simulation settings.
    controller_type:   ``ControllerType.FIXED_TIME`` or ``ControllerType.ADAPTIVE``.
    schedule:          the arrival plan. ``None`` draws a new one from ``config``.
                       To compare controllers, pass the SAME schedule to both
                       (or simply use ``run_comparison``). It must cover
                       ``config.duration`` seconds.
    adaptive_options:  optional settings for the adaptive controller only
                       (``starvation_threshold``, ``queue_weight``, ``wait_weight``).

    Usage
    -----
        result = Simulator(config, ControllerType.ADAPTIVE, schedule).run()

    or second by second (handy for live animations):

        sim = Simulator(config, ControllerType.FIXED_TIME)
        while not sim.is_finished:
            record = sim.step()

    ``run()`` always starts from a fresh state (``reset()``), so running twice
    gives identical results.
    """

    def __init__(
        self,
        config: SimulationConfig,
        controller_type: ControllerType,
        schedule: Optional[TrafficSchedule] = None,
        adaptive_options: Optional[Mapping] = None,
    ) -> None:
        if not isinstance(config, SimulationConfig):
            raise ValueError(f"config must be a SimulationConfig, got {config!r}")
        if not isinstance(controller_type, ControllerType):
            raise ValueError(f"controller_type must be a ControllerType, got {controller_type!r}")

        if schedule is None:
            schedule = TrafficSchedule.from_config(config)
        elif not isinstance(schedule, TrafficSchedule):
            raise ValueError(f"schedule must be a TrafficSchedule, got {schedule!r}")
        if schedule.duration != config.duration:
            raise ValueError(
                f"schedule covers {schedule.duration} seconds but config.duration is {config.duration}"
            )

        options: Dict[str, Any] = {}
        if adaptive_options is not None:
            if not isinstance(adaptive_options, Mapping):
                raise ValueError(f"adaptive_options must be a mapping, got {adaptive_options!r}")
            if controller_type is not ControllerType.ADAPTIVE:
                raise ValueError("adaptive_options can only be used with ControllerType.ADAPTIVE")
            unknown = sorted(set(adaptive_options) - set(_ADAPTIVE_OPTIONS))
            if unknown:
                raise ValueError(f"unknown adaptive option(s): {unknown}; allowed: {list(_ADAPTIVE_OPTIONS)}")
            options = dict(adaptive_options)

        self._config: SimulationConfig = config
        self._controller_type: ControllerType = controller_type
        self._schedule: TrafficSchedule = schedule
        self._adaptive_options: Dict[str, Any] = options
        # Exact fraction, so 0.3 vehicles/second really is 3 vehicles per 10 seconds.
        self._discharge_rate: Fraction = Fraction(config.discharge_rate).limit_denominator(1_000_000)

        self.reset()

    # ----- read-only information ---------------------------------------------
    @property
    def config(self) -> SimulationConfig:
        return self._config

    @property
    def schedule(self) -> TrafficSchedule:
        return self._schedule

    @property
    def controller_type(self) -> ControllerType:
        return self._controller_type

    @property
    def controller(self) -> TrafficController:
        return self._controller

    @property
    def junction(self) -> Junction:
        return self._junction

    @property
    def lanes(self) -> Dict[Direction, Lane]:
        """The four lanes of the CURRENT run (a copy of the dictionary)."""
        return dict(self._lanes)

    @property
    def time(self) -> int:
        """The next second to be simulated (= number of seconds simulated so far)."""
        return self._time

    @property
    def is_finished(self) -> bool:
        return self._time >= self._config.duration

    # ----- starting over -------------------------------------------------------
    def reset(self) -> None:
        """Create a brand-new junction, lanes and controller and go back to second 0."""
        self._lanes: Dict[Direction, Lane] = {d: Lane(d) for d in _DIRECTIONS}
        self._junction: Junction = Junction(self._config)
        if self._controller_type is ControllerType.FIXED_TIME:
            self._controller: TrafficController = FixedTimeController(self._junction)
        else:
            self._controller = AdaptiveController(self._junction, self._lanes, **self._adaptive_options)

        self._time: int = 0
        self._history: List[StepRecord] = []
        self._vehicles: List[Tuple[Direction, Vehicle]] = []
        self._credit: Dict[Direction, Fraction] = {d: Fraction(0) for d in _DIRECTIONS}
        self._total_arrived: int = 0
        self._total_departed: int = 0

    # ----- running ---------------------------------------------------------------
    def run(self) -> SimulationResult:
        """Run the whole simulation from a fresh state and return the result."""
        self.reset()
        while not self.is_finished:
            self.step()
        return self.result()

    def step(self) -> StepRecord:
        """Simulate exactly ONE second (see the module docstring for the order)."""
        if self.is_finished:
            raise RuntimeError("the simulation has already finished; call reset() to start again")

        t = self._time
        arrived = self._arrive(t)                       # 1. ARRIVE
        decision = self._decide(t)                      # 2. DECIDE
        if not self._junction.is_safe():                # defence in depth: never trust, always check
            raise RuntimeError(f"unsafe signals at t={t}: {self._junction.signal_states()}")
        departed = self._depart(t)                      # 3. DEPART (signals of second t)

        self._total_arrived += sum(arrived.values())
        self._total_departed += sum(departed.values())
        record = StepRecord(                            # 4. RECORD
            time=t,
            stage=self._junction.stage,
            active_phase=self._junction.active_phase,
            signals=self._junction.signal_states(),
            queue_lengths={d: self._lanes[d].queue_length for d in _DIRECTIONS},
            arrived=arrived,
            departed=departed,
            total_arrived=self._total_arrived,
            total_departed=self._total_departed,
            decision=decision,
        )
        self._history.append(record)

        self._junction.advance()                        # 5. ADVANCE
        self._time += 1
        return record

    def result(self) -> SimulationResult:
        """Snapshot of everything simulated so far (complete after ``run()``)."""
        vehicles = tuple(
            VehicleRecord(
                vehicle_id=vehicle.vehicle_id,
                direction=direction,
                arrival_time=vehicle.arrival_time,
                departure_time=vehicle.departure_time,
            )
            for direction, vehicle in self._vehicles
        )
        return SimulationResult(
            config=self._config,
            controller_type=self._controller_type,
            controller_name=self._controller.name,
            history=tuple(self._history),
            vehicles=vehicles,
        )

    # ----- the five steps (private) ----------------------------------------------
    def _arrive(self, t: int) -> Dict[Direction, int]:
        """Create this second's vehicles from the schedule and queue them in their lanes."""
        arrived: Dict[Direction, int] = {}
        for direction in _DIRECTIONS:
            ids = self._schedule.vehicle_ids(t, direction)
            for vehicle_id in ids:
                vehicle = Vehicle(vehicle_id=vehicle_id, arrival_time=t)
                self._lanes[direction].add_vehicle(vehicle)
                self._vehicles.append((direction, vehicle))
            arrived[direction] = len(ids)
        return arrived

    def _decide(self, t: int) -> ControllerDecision:
        """Let the controller decide, and make sure it only used its own interface."""
        counters_before = self._lane_counters()
        elapsed_before = self._junction.seconds_in_stage

        decision = self._controller.update(t)

        if self._lane_counters() != counters_before:
            raise RuntimeError("a controller must not add or remove vehicles")
        if self._junction.seconds_in_stage != elapsed_before:
            raise RuntimeError("a controller must not advance the junction; only the simulator does")
        return decision

    def _lane_counters(self) -> Tuple[Tuple[int, int, int], ...]:
        return tuple(
            (lane.total_arrived, lane.total_departed, lane.queue_length) for lane in self._lanes.values()
        )

    def _depart(self, t: int) -> Dict[Direction, int]:
        """Discharge vehicles. ONLY green directions may release vehicles."""
        green = set(self._junction.green_directions())
        departed: Dict[Direction, int] = {}
        for direction in _DIRECTIONS:
            lane = self._lanes[direction]
            if direction not in green:
                self._credit[direction] = Fraction(0)    # red or yellow: no capacity is earned
                departed[direction] = 0
                continue

            credit = self._credit[direction] + self._discharge_rate
            count = 0
            while credit >= 1 and not lane.is_empty():
                lane.remove_vehicle(t)
                credit -= 1
                count += 1
            if lane.is_empty():
                credit = Fraction(0)                     # unused capacity is lost
            self._credit[direction] = credit
            departed[direction] = count
        return departed


# --------------------------------------------------------------------------- #
# Convenience: the fair comparison in one call
# --------------------------------------------------------------------------- #
def run_comparison(
    config: SimulationConfig,
    schedule: Optional[TrafficSchedule] = None,
    adaptive_options: Optional[Mapping] = None,
) -> ComparisonResult:
    """Run the fixed-time AND the adaptive controller on the SAME traffic schedule.

    The schedule is drawn once (unless you supply one) and shared by both runs.
    """
    if schedule is None:
        schedule = TrafficSchedule.from_config(config)
    fixed_time = Simulator(config, ControllerType.FIXED_TIME, schedule).run()
    adaptive = Simulator(config, ControllerType.ADAPTIVE, schedule, adaptive_options).run()
    return ComparisonResult(schedule=schedule, fixed_time=fixed_time, adaptive=adaptive)
