"""
metrics.py
==========
Measures how well a traffic controller performed. It reads the finished
``SimulationResult`` produced by simulator.py and turns it into numbers.

This module contains NO plotting, NO user interface and NO randomness.
The same result always gives exactly the same metrics.

Conventions (taken from simulator.py, never invented here)
----------------------------------------------------------
* T = ``len(result.history)`` = number of seconds actually simulated.
  For a complete run T equals ``config.duration``.
* ``StepRecord.queue_lengths[d]`` is the queue of lane d at the END of a second.
  Every queue metric below samples the queues at those T moments.
* A vehicle's waiting time is ``departure_time - arrival_time``.
  A vehicle that is still queued when the run stops has not finished waiting,
  so it is measured up to T (``VehicleRecord.waiting_time(T)``).

Metric definitions
------------------
 1. average_waiting_time        = sum of waits of ALL arrived vehicles / vehicles arrived
                                  (vehicles still queued are included, measured up to T)
    average_waiting_time_departed = same, but only vehicles that have already left
 2. max_waiting_time            = largest single waiting time (same rule for queued vehicles)
 3. average_queue_length        = (sum over seconds and lanes of queue) / (T * 4 lanes)
                                  -> the typical queue of ONE lane
    average_total_queue_length  = (sum over seconds of all four queues) / T
 4. average_queue_by_direction  = (sum over seconds of that lane's queue) / T
 5. max_queue_length            = longest queue of any single lane at any second
    max_queue_by_direction      = longest queue of each lane
    max_total_queue_length      = largest number of vehicles waiting in all lanes at once
 6. total_arrived / total_departed / vehicles_remaining
                                  (remaining = arrived - departed = queues at the end)
 7. throughput_per_minute       = vehicles departed * 60 / T
                                  (measured over the whole run, including yellow and all-red)
 8. congestion_index            = lane-seconds with queue >= threshold / (T * 4)
                                  a number between 0 and 1: the share of the time
                                  lanes spent "congested" (default threshold: 10 vehicles)
    unserved_ratio              = vehicles_remaining / total_arrived
                                  (share of the demand NOT served by the end)
 9. percentage improvement      = see ``percentage_improvement``

No double counting
------------------
Each vehicle's wait is computed ONCE from its two timestamps. Nothing is
added up second by second. As an independent check, the total waiting time
(vehicle-seconds) must equal the sum of all queue lengths over time; the
tests verify that identity.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from numbers import Integral, Real
from typing import Any, Dict, List, Optional, Tuple

from config import Direction, SimulationConfig
from simulator import (
    ComparisonResult,
    ControllerType,
    SimulationResult,
    StepRecord,
    VehicleRecord,
)

_DIRECTIONS: Tuple[Direction, ...] = tuple(Direction)
SECONDS_PER_MINUTE: int = 60

#: A lane counts as "congested" in a second when it holds at least this many vehicles.
#: This is an assumption of OUR project (not a law of nature); it can be changed per call.
DEFAULT_CONGESTION_THRESHOLD: int = 10

# Two values closer than this are treated as equal when deciding a winner (float noise).
_TOLERANCE: float = 1e-9


# --------------------------------------------------------------------------- #
# Small helpers (private)
# --------------------------------------------------------------------------- #
def _whole(name: str, value: Any, minimum: int = 0) -> int:
    """Return ``value`` as int, or raise ValueError unless it is a whole number >= minimum."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a whole number, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value!r}")
    return int(value)


def _finite(name: str, value: Any) -> float:
    """Return ``value`` as float, or raise ValueError unless it is a finite real number."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a number, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return float(value)


def _divide(numerator: float, denominator: float) -> float:
    """numerator / denominator, but 0.0 when the denominator is 0 (e.g. an empty run)."""
    return numerator / denominator if denominator else 0.0


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SimulationMetrics:
    """All metrics of ONE simulation run. Treat the dictionaries as read-only."""

    # --- which run is this? ---
    controller_type: ControllerType
    controller_name: str
    duration: int                 # configured length (seconds)
    steps_completed: int          # seconds actually simulated (T)
    is_complete: bool

    # --- vehicle accounting ---
    total_arrived: int
    total_departed: int
    vehicles_remaining: int
    arrived_by_direction: Dict[Direction, int]
    departed_by_direction: Dict[Direction, int]

    # --- waiting time (seconds) ---
    total_waiting_time: int                  # sum of all waits, in vehicle-seconds
    average_waiting_time: float              # all arrived vehicles
    average_waiting_time_departed: float     # only vehicles that have left
    max_waiting_time: int

    # --- queues (vehicles) ---
    average_queue_length: float              # per lane
    average_total_queue_length: float        # all four lanes together
    average_queue_by_direction: Dict[Direction, float]
    max_queue_length: int                    # worst single lane
    max_queue_by_direction: Dict[Direction, int]
    max_total_queue_length: int

    # --- flow and congestion ---
    throughput_per_minute: float
    congestion_threshold: int
    congestion_index: float                  # 0.0 .. 1.0
    unserved_ratio: float                    # 0.0 .. 1.0


class Winner(Enum):
    """Which controller did better on one metric."""

    ADAPTIVE = "adaptive"
    FIXED_TIME = "fixed_time"
    TIE = "tie"


@dataclass(frozen=True)
class MetricComparison:
    """One metric, for both controllers, plus the percentage improvement.

    ``improvement_percent`` is positive when the adaptive controller is better,
    negative when it is worse, 0.0 when equal, and ``None`` when it cannot be
    defined (the fixed-time baseline is 0 but the adaptive value is not).
    """

    name: str
    label: str
    unit: str
    higher_is_better: bool
    fixed_time: float
    adaptive: float
    improvement_percent: Optional[float]

    @property
    def winner(self) -> Winner:
        """Decided from the raw values (not from the percentage), so it is always defined."""
        if math.isclose(self.fixed_time, self.adaptive, rel_tol=_TOLERANCE, abs_tol=_TOLERANCE):
            return Winner.TIE
        adaptive_is_higher = self.adaptive > self.fixed_time
        adaptive_is_better = adaptive_is_higher == self.higher_is_better
        return Winner.ADAPTIVE if adaptive_is_better else Winner.FIXED_TIME


@dataclass(frozen=True)
class MetricsComparison:
    """Metrics of both controllers on the same traffic, side by side."""

    fixed_time: SimulationMetrics
    adaptive: SimulationMetrics
    rows: Tuple[MetricComparison, ...]

    def row(self, name: str) -> MetricComparison:
        """Look up one comparison row by metric name, e.g. ``"average_waiting_time"``."""
        for row in self.rows:
            if row.name == name:
                return row
        raise ValueError(f"unknown metric {name!r}; available: {[r.name for r in self.rows]}")

    def count(self, winner: Winner) -> int:
        """How many compared metrics were won by ``winner``."""
        if not isinstance(winner, Winner):
            raise ValueError(f"winner must be a Winner, got {winner!r}")
        return sum(1 for row in self.rows if row.winner is winner)

    def as_records(self) -> List[Dict[str, Any]]:
        """Plain list of dictionaries (one per metric), ready for a table or a DataFrame."""
        return [
            {
                "metric": row.name,
                "label": row.label,
                "unit": row.unit,
                "fixed_time": row.fixed_time,
                "adaptive": row.adaptive,
                "improvement_percent": row.improvement_percent,
                "winner": row.winner.value,
            }
            for row in self.rows
        ]


# (attribute name, label, unit, higher_is_better) of every compared metric
_COMPARED_METRICS: Tuple[Tuple[str, str, str, bool], ...] = (
    ("average_waiting_time", "Average waiting time", "s", False),
    ("max_waiting_time", "Maximum waiting time", "s", False),
    ("average_queue_length", "Average queue length (per lane)", "vehicles", False),
    ("max_queue_length", "Maximum queue length (single lane)", "vehicles", False),
    ("vehicles_remaining", "Vehicles remaining in queues", "vehicles", False),
    ("throughput_per_minute", "Throughput", "vehicles/min", True),
    ("congestion_index", "Congestion index", "fraction of lane-seconds", False),
)


# --------------------------------------------------------------------------- #
# Validation of a SimulationResult
# --------------------------------------------------------------------------- #
def _check_direction_counts(where: str, name: str, mapping: Any) -> None:
    """The mapping must hold one whole number >= 0 for each of the four directions."""
    if not isinstance(mapping, Mapping) or set(mapping) != set(_DIRECTIONS):
        raise ValueError(f"{where}: {name} must contain exactly the four directions")
    for direction in _DIRECTIONS:
        _whole(f"{where}: {name}[{direction.name}]", mapping[direction])


def _check_result(result: Any) -> None:
    """Raise ValueError unless the result is internally consistent.

    Checks that every vehicle is sensible and that the per-second history agrees
    with the vehicle list (arrivals, departures and queue lengths all balance).
    """
    if not isinstance(result, SimulationResult):
        raise ValueError(f"result must be a SimulationResult, got {result!r}")
    if not isinstance(result.config, SimulationConfig):
        raise ValueError("result.config must be a SimulationConfig")
    if not isinstance(result.controller_type, ControllerType):
        raise ValueError("result.controller_type must be a ControllerType")

    steps = len(result.history)
    if steps > result.config.duration:
        raise ValueError(f"history has {steps} seconds but duration is {result.config.duration}")

    # ---- vehicles ----
    seen_ids = set()
    arrivals: Counter = Counter()      # (second, direction) -> vehicles that arrived
    departures: Counter = Counter()    # (second, direction) -> vehicles that left
    for vehicle in result.vehicles:
        if not isinstance(vehicle, VehicleRecord):
            raise ValueError(f"vehicles must hold VehicleRecord objects, got {vehicle!r}")
        _whole("vehicle_id", vehicle.vehicle_id, minimum=1)
        if vehicle.vehicle_id in seen_ids:
            raise ValueError(f"duplicate vehicle_id {vehicle.vehicle_id}")
        seen_ids.add(vehicle.vehicle_id)
        if not isinstance(vehicle.direction, Direction):
            raise ValueError(f"vehicle {vehicle.vehicle_id}: direction must be a Direction")

        arrival = _whole(f"vehicle {vehicle.vehicle_id}: arrival_time", vehicle.arrival_time)
        if arrival >= steps:
            raise ValueError(
                f"vehicle {vehicle.vehicle_id}: arrival_time {arrival} is outside the "
                f"{steps} simulated seconds"
            )
        arrivals[(arrival, vehicle.direction)] += 1

        if vehicle.departure_time is not None:
            departure = _whole(f"vehicle {vehicle.vehicle_id}: departure_time", vehicle.departure_time)
            if departure < arrival:
                raise ValueError(f"vehicle {vehicle.vehicle_id}: departs before it arrives")
            if departure >= steps:
                raise ValueError(
                    f"vehicle {vehicle.vehicle_id}: departure_time {departure} is outside the "
                    f"{steps} simulated seconds"
                )
            departures[(departure, vehicle.direction)] += 1

    # ---- history ----
    queue = {direction: 0 for direction in _DIRECTIONS}
    running_arrived = 0
    running_departed = 0
    for index, record in enumerate(result.history):
        if not isinstance(record, StepRecord):
            raise ValueError(f"history must hold StepRecord objects, got {record!r}")
        where = f"second {index}"
        if record.time != index:
            raise ValueError(f"{where}: history is out of order (record says time {record.time!r})")
        _check_direction_counts(where, "queue_lengths", record.queue_lengths)
        _check_direction_counts(where, "arrived", record.arrived)
        _check_direction_counts(where, "departed", record.departed)

        for direction in _DIRECTIONS:
            if record.arrived[direction] != arrivals[(index, direction)]:
                raise ValueError(f"{where}: arrivals of {direction.name} do not match the vehicle list")
            if record.departed[direction] != departures[(index, direction)]:
                raise ValueError(f"{where}: departures of {direction.name} do not match the vehicle list")
            queue[direction] += record.arrived[direction] - record.departed[direction]
            if record.queue_lengths[direction] != queue[direction]:
                raise ValueError(
                    f"{where}: queue of {direction.name} is {record.queue_lengths[direction]} "
                    f"but arrivals minus departures give {queue[direction]}"
                )
            running_arrived += record.arrived[direction]
            running_departed += record.departed[direction]

        if record.total_arrived != running_arrived or record.total_departed != running_departed:
            raise ValueError(f"{where}: cumulative totals do not match the per-second counts")


# --------------------------------------------------------------------------- #
# Calculation
# --------------------------------------------------------------------------- #
def calculate_metrics(
    result: SimulationResult,
    congestion_threshold: int = DEFAULT_CONGESTION_THRESHOLD,
) -> SimulationMetrics:
    """Compute every metric of one run (complete, partial or empty).

    Raises ValueError for a bad threshold or an inconsistent result.
    """
    threshold = _whole("congestion_threshold", congestion_threshold, minimum=1)
    _check_result(result)

    steps = len(result.history)   # T

    # ---- vehicles: each vehicle is counted exactly once ----
    arrived_by = {direction: 0 for direction in _DIRECTIONS}
    departed_by = {direction: 0 for direction in _DIRECTIONS}
    total_wait = 0
    wait_of_departed = 0
    max_wait = 0
    for vehicle in result.vehicles:
        wait = vehicle.waiting_time(steps)   # departed: fixed; still queued: measured up to T
        total_wait += wait
        max_wait = max(max_wait, wait)
        arrived_by[vehicle.direction] += 1
        if vehicle.has_departed:
            departed_by[vehicle.direction] += 1
            wait_of_departed += wait

    total_arrived = len(result.vehicles)
    total_departed = sum(departed_by.values())
    remaining = total_arrived - total_departed

    # ---- queues: sampled at the end of every simulated second ----
    queue_sum = {direction: 0 for direction in _DIRECTIONS}
    queue_max = {direction: 0 for direction in _DIRECTIONS}
    max_total = 0
    congested_lane_seconds = 0
    for record in result.history:
        second_total = 0
        for direction in _DIRECTIONS:
            length = record.queue_lengths[direction]
            queue_sum[direction] += length
            queue_max[direction] = max(queue_max[direction], length)
            second_total += length
            if length >= threshold:
                congested_lane_seconds += 1
        max_total = max(max_total, second_total)

    lane_count = len(_DIRECTIONS)
    all_queue_sum = sum(queue_sum.values())

    return SimulationMetrics(
        controller_type=result.controller_type,
        controller_name=result.controller_name,
        duration=result.config.duration,
        steps_completed=steps,
        is_complete=(steps == result.config.duration),
        total_arrived=total_arrived,
        total_departed=total_departed,
        vehicles_remaining=remaining,
        arrived_by_direction=arrived_by,
        departed_by_direction=departed_by,
        total_waiting_time=total_wait,
        average_waiting_time=_divide(total_wait, total_arrived),
        average_waiting_time_departed=_divide(wait_of_departed, total_departed),
        max_waiting_time=max_wait,
        average_queue_length=_divide(all_queue_sum, steps * lane_count),
        average_total_queue_length=_divide(all_queue_sum, steps),
        average_queue_by_direction={d: _divide(queue_sum[d], steps) for d in _DIRECTIONS},
        max_queue_length=max(queue_max.values()),
        max_queue_by_direction=queue_max,
        max_total_queue_length=max_total,
        throughput_per_minute=_divide(total_departed * SECONDS_PER_MINUTE, steps),
        congestion_threshold=threshold,
        congestion_index=_divide(congested_lane_seconds, steps * lane_count),
        unserved_ratio=_divide(remaining, total_arrived),
    )


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #
def percentage_improvement(
    baseline: float,
    candidate: float,
    higher_is_better: bool = False,
) -> Optional[float]:
    """How much better (in %) the candidate is than the baseline.

    Lower is better  (waiting time, queues ...):  (baseline - candidate) / baseline * 100
    Higher is better (throughput ...):            (candidate - baseline) / baseline * 100

    Positive = candidate is better, negative = candidate is worse, 0.0 = equal.
    If the baseline is 0 the percentage is undefined: the result is 0.0 when the
    candidate is also 0, otherwise ``None`` (we refuse to invent a number).
    """
    baseline = _finite("baseline", baseline)
    candidate = _finite("candidate", candidate)
    if not isinstance(higher_is_better, bool):
        raise ValueError(f"higher_is_better must be True or False, got {higher_is_better!r}")

    if baseline == 0:
        return 0.0 if candidate == 0 else None
    change = (candidate - baseline) if higher_is_better else (baseline - candidate)
    return change / abs(baseline) * 100.0


def compare_metrics(fixed_time: SimulationMetrics, adaptive: SimulationMetrics) -> MetricsComparison:
    """Compare the metrics of the fixed-time run with those of the adaptive run.

    Both must describe the same traffic: same number of simulated seconds, same
    arrivals per direction and the same congestion threshold.
    """
    if not isinstance(fixed_time, SimulationMetrics):
        raise ValueError(f"fixed_time must be SimulationMetrics, got {fixed_time!r}")
    if not isinstance(adaptive, SimulationMetrics):
        raise ValueError(f"adaptive must be SimulationMetrics, got {adaptive!r}")
    if fixed_time.controller_type is not ControllerType.FIXED_TIME:
        raise ValueError("the first argument must be the FIXED_TIME run")
    if adaptive.controller_type is not ControllerType.ADAPTIVE:
        raise ValueError("the second argument must be the ADAPTIVE run")
    if fixed_time.steps_completed != adaptive.steps_completed:
        raise ValueError("both runs must cover the same number of simulated seconds")
    if fixed_time.arrived_by_direction != adaptive.arrived_by_direction:
        raise ValueError("both runs must have the same arrivals (same traffic schedule)")
    if fixed_time.congestion_threshold != adaptive.congestion_threshold:
        raise ValueError("both runs must use the same congestion threshold")

    rows = []
    for name, label, unit, higher_is_better in _COMPARED_METRICS:
        fixed_value = float(getattr(fixed_time, name))
        adaptive_value = float(getattr(adaptive, name))
        rows.append(
            MetricComparison(
                name=name,
                label=label,
                unit=unit,
                higher_is_better=higher_is_better,
                fixed_time=fixed_value,
                adaptive=adaptive_value,
                improvement_percent=percentage_improvement(fixed_value, adaptive_value, higher_is_better),
            )
        )
    return MetricsComparison(fixed_time=fixed_time, adaptive=adaptive, rows=tuple(rows))


def compare_results(
    fixed_time: SimulationResult,
    adaptive: SimulationResult,
    congestion_threshold: int = DEFAULT_CONGESTION_THRESHOLD,
) -> MetricsComparison:
    """Compute and compare the metrics of two runs, after checking the comparison is fair.

    Fair means: same configuration and exactly the same vehicles (ID, direction,
    arrival second). Only the controller may differ.
    """
    fixed_metrics = calculate_metrics(fixed_time, congestion_threshold)   # also validates
    adaptive_metrics = calculate_metrics(adaptive, congestion_threshold)

    if fixed_time.config != adaptive.config:
        raise ValueError("both runs must use the same SimulationConfig")

    def traffic(result: SimulationResult) -> List[Tuple[int, str, int]]:
        return sorted((v.vehicle_id, v.direction.name, v.arrival_time) for v in result.vehicles)

    if traffic(fixed_time) != traffic(adaptive):
        raise ValueError("both runs must receive exactly the same vehicles (same traffic schedule)")

    return compare_metrics(fixed_metrics, adaptive_metrics)


def evaluate_comparison(
    comparison: ComparisonResult,
    congestion_threshold: int = DEFAULT_CONGESTION_THRESHOLD,
) -> MetricsComparison:
    """Convenience: metrics and comparison for the output of ``run_comparison``."""
    if not isinstance(comparison, ComparisonResult):
        raise ValueError(f"comparison must be a ComparisonResult, got {comparison!r}")
    return compare_results(comparison.fixed_time, comparison.adaptive, congestion_threshold)
