"""Pandas and CSV views of existing SmartFlow AI simulation data.

This module only converts already-produced simulation results and metrics.
It does not run simulations, generate traffic, or calculate metrics.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import Any

import pandas as pd

from config import Direction
from metrics import MetricsComparison, SimulationMetrics
from simulator import SimulationResult, StepRecord


_COMPARISON_COLUMNS = (
    "metric",
    "label",
    "unit",
    "fixed_time",
    "adaptive",
    "improvement_percent",
    "winner",
)

_DIRECTION_NAMES = ("north", "south", "east", "west")
_HISTORY_COLUMNS = (
    "time",
    "stage",
    "active_phase",
    "total_arrived",
    "total_departed",
    "total_waiting",
    "north_signal",
    "north_queue",
    "south_signal",
    "south_queue",
    "east_signal",
    "east_queue",
    "west_signal",
    "west_queue",
    "north_arrived",
    "north_departed",
    "south_arrived",
    "south_departed",
    "east_arrived",
    "east_departed",
    "west_arrived",
    "west_departed",
)
_DIRECTION_COLUMNS = (
    "controller_name",
    "direction",
    "vehicles_arrived",
    "vehicles_departed",
    "average_queue_length",
    "maximum_queue_length",
)


def _enum_value(value: Any) -> Any:
    """Return an enum's public value, or preserve a non-enum value."""
    return value.value if isinstance(value, Enum) else value


def _require_metrics_comparison(value: Any) -> MetricsComparison:
    if not isinstance(value, MetricsComparison):
        raise ValueError(
            f"metrics_comparison must be a MetricsComparison, got {value!r}"
        )
    return value


def _require_result(value: Any) -> SimulationResult:
    if not isinstance(value, SimulationResult):
        raise ValueError(f"result must be a SimulationResult, got {value!r}")
    return value


def _require_direction_mapping(metrics: SimulationMetrics, name: str) -> Mapping:
    mapping = getattr(metrics, name, None)
    if not isinstance(mapping, Mapping):
        raise ValueError(
            f"{metrics.controller_name}: {name} must be a mapping, got {mapping!r}"
        )
    if set(mapping) != set(Direction):
        raise ValueError(
            f"{metrics.controller_name}: {name} must contain exactly North, South, East, and West"
        )
    return mapping


def comparison_metrics_frame(metrics_comparison: MetricsComparison) -> pd.DataFrame:
    """Convert a MetricsComparison.as_records() result to a DataFrame."""
    comparison = _require_metrics_comparison(metrics_comparison)
    records = comparison.as_records()
    if not isinstance(records, list):
        raise ValueError("MetricsComparison.as_records() must return a list")
    return pd.DataFrame(records, columns=list(_COMPARISON_COLUMNS))


def direction_metrics_frame(metrics_comparison: MetricsComparison) -> pd.DataFrame:
    """Build one row per controller and direction for directional metrics."""
    comparison = _require_metrics_comparison(metrics_comparison)

    rows = []
    for metrics in (comparison.fixed_time, comparison.adaptive):
        if not isinstance(metrics, SimulationMetrics):
            raise ValueError(
                f"{metrics!r} must be a SimulationMetrics"
            )

        arrived = _require_direction_mapping(metrics, "arrived_by_direction")
        departed = _require_direction_mapping(metrics, "departed_by_direction")
        average_queue = _require_direction_mapping(metrics, "average_queue_by_direction")
        maximum_queue = _require_direction_mapping(metrics, "max_queue_by_direction")

        for direction in Direction:
            rows.append(
                {
                    "controller_name": metrics.controller_name,
                    "direction": direction.value,
                    "vehicles_arrived": arrived[direction],
                    "vehicles_departed": departed[direction],
                    "average_queue_length": average_queue[direction],
                    "maximum_queue_length": maximum_queue[direction],
                }
            )

    return pd.DataFrame(rows, columns=list(_DIRECTION_COLUMNS))


def history_frame(result: SimulationResult) -> pd.DataFrame:
    """Convert a SimulationResult history into one row per simulated second."""
    result = _require_result(result)

    rows = []
    for record in result.history:
        if not isinstance(record, StepRecord):
            raise ValueError(
                f"history entries must be StepRecord objects, got {record!r}"
            )

        signals = record.signals if isinstance(record.signals, Mapping) else {}
        queues = record.queue_lengths if isinstance(record.queue_lengths, Mapping) else {}
        arrivals = record.arrived if isinstance(record.arrived, Mapping) else {}
        departures = record.departed if isinstance(record.departed, Mapping) else {}

        row: dict[str, Any] = {
            "time": record.time,
            "stage": _enum_value(record.stage),
            "active_phase": (
                pd.NA if record.active_phase is None else _enum_value(record.active_phase)
            ),
            "total_arrived": record.total_arrived,
            "total_departed": record.total_departed,
            "total_waiting": record.total_waiting,
        }

        for direction, name in zip(Direction, _DIRECTION_NAMES):
            signal = signals.get(direction, pd.NA)
            queue = queues.get(direction, pd.NA)
            arrived = arrivals.get(direction, pd.NA)
            departed = departures.get(direction, pd.NA)
            row[f"{name}_signal"] = _enum_value(signal)
            row[f"{name}_queue"] = queue
            row[f"{name}_arrived"] = arrived
            row[f"{name}_departed"] = departed

        rows.append(row)

    return pd.DataFrame(rows, columns=list(_HISTORY_COLUMNS))


def history_csv_bytes(result: SimulationResult) -> bytes:
    """Return the history table as UTF-8 encoded CSV bytes."""
    frame = history_frame(result)
    return frame.to_csv(index=False).encode("utf-8")


def metrics_csv_bytes(metrics_comparison: MetricsComparison) -> bytes:
    """Return the comparison metrics table as UTF-8 encoded CSV bytes."""
    frame = comparison_metrics_frame(metrics_comparison)
    return frame.to_csv(index=False).encode("utf-8")


def direction_csv_bytes(metrics_comparison: MetricsComparison) -> bytes:
    """Return the directional metrics table as UTF-8 encoded CSV bytes."""
    frame = direction_metrics_frame(metrics_comparison)
    return frame.to_csv(index=False).encode("utf-8")
