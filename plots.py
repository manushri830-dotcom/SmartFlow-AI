"""
plots.py
========
Charts for SmartFlow AI. This module ONLY draws data that already exists:

    * ``MetricsComparison``  (from metrics.py)   -> bar charts
    * ``SimulationResult``   (from simulator.py) -> time-series charts

It never runs a simulation, never creates traffic, never touches queues or
signals and never recalculates metrics. Every number on a chart was produced
by simulator.py or metrics.py.

Design rules
------------
* Every function RETURNS a ``matplotlib.figure.Figure`` and shows nothing.
  A Streamlit page can display it with ``st.pyplot(fig)``; a script can save it
  with ``fig.savefig(...)`` or ``figure_to_png_bytes(fig)``.
* Figures are created with ``Figure()`` directly, NOT with ``pyplot``. So no
  global figure registry is used, no window is ever opened, no backend has to
  be chosen, and nothing needs ``plt.close()``: when you drop the reference,
  the figure is garbage-collected.
* Bar charts always start at 0 (a truncated axis makes small differences look
  huge). Time-series charts also start at 0.
* Empty or unavailable data gives a figure with a short message, not a crash.
  WRONG TYPES raise ``ValueError`` (same convention as the rest of the project).
* The same colours are used for the same things in every chart:
  Fixed-Time = blue, Adaptive = orange, signals green / yellow / red.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from io import BytesIO
from numbers import Integral, Real
from typing import Any, Callable, Dict, List, Optional, Tuple

from matplotlib.figure import Figure
from matplotlib.patches import Patch

from config import Direction
from junction import SignalState
from metrics import MetricComparison, MetricsComparison, Winner
from simulator import ComparisonResult, ControllerType, SimulationResult

_DIRECTIONS: Tuple[Direction, ...] = tuple(Direction)

# --------------------------------------------------------------------------- #
# Style (one place, so every chart looks the same)
# --------------------------------------------------------------------------- #
FIXED_COLOR: str = "#0072B2"      # blue   (colour-blind friendly palette)
ADAPTIVE_COLOR: str = "#E69F00"   # orange
DIRECTION_COLORS: Dict[Direction, str] = {
    Direction.NORTH: "#0072B2",
    Direction.SOUTH: "#009E73",
    Direction.EAST: "#D55E00",
    Direction.WEST: "#CC79A7",
}
SIGNAL_COLORS: Dict[SignalState, str] = {
    SignalState.GREEN: "#2E9E4F",
    SignalState.YELLOW: "#F2C230",
    SignalState.RED: "#D64545",
}
SIGNAL_LABELS: Dict[SignalState, str] = {
    SignalState.GREEN: "Green",
    SignalState.YELLOW: "Yellow",
    SignalState.RED: "Red",
}

FIGSIZE_BARS: Tuple[float, float] = (7.0, 4.8)
FIGSIZE_GROUPED: Tuple[float, float] = (8.0, 4.8)
FIGSIZE_TIME: Tuple[float, float] = (10.0, 4.8)
FIGSIZE_SIGNALS: Tuple[float, float] = (10.0, 3.8)

_UNIT_NAMES: Dict[str, str] = {"s": "seconds"}
_NO_DATA = "No simulation data to display."
_NO_QUEUE_DATA = "Queue data is not available in this simulation history."
_NO_SIGNAL_DATA = "Signal data is not available in this simulation history."
_NO_FLOW_DATA = "Arrival/departure data is not available in this simulation history."


# --------------------------------------------------------------------------- #
# Validation helpers (private)
# --------------------------------------------------------------------------- #
def _require_comparison(value: Any) -> MetricsComparison:
    if not isinstance(value, MetricsComparison):
        raise ValueError(f"comparison must be a MetricsComparison, got {value!r}")
    return value


def _require_result(name: str, value: Any) -> SimulationResult:
    if not isinstance(value, SimulationResult):
        raise ValueError(f"{name} must be a SimulationResult, got {value!r}")
    return value


def _finite(name: str, value: Any) -> float:
    """Return value as float, or raise ValueError unless it is a finite number."""
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return float(value)


def _direction_values(name: str, mapping: Any) -> List[float]:
    """Values of a {Direction: number} dictionary in the fixed N, S, E, W order."""
    if not isinstance(mapping, Mapping) or set(mapping) != set(_DIRECTIONS):
        raise ValueError(f"{name} must contain exactly the four directions")
    values = [_finite(f"{name}[{d.name}]", mapping[d]) for d in _DIRECTIONS]
    if any(v < 0 for v in values):
        raise ValueError(f"{name} must not contain negative values")
    return values


def _check_window(start: Any, end: Any) -> None:
    """start: whole second >= 0. end: None or a whole second > start (exclusive)."""
    if isinstance(start, bool) or not isinstance(start, Integral) or start < 0:
        raise ValueError(f"start must be a whole number of seconds >= 0, got {start!r}")
    if end is not None:
        if isinstance(end, bool) or not isinstance(end, Integral) or end <= start:
            raise ValueError(f"end must be a whole number greater than start ({start}), got {end!r}")


# --------------------------------------------------------------------------- #
# Drawing helpers (private)
# --------------------------------------------------------------------------- #
def _new_axes(figsize: Tuple[float, float]):
    figure = Figure(figsize=figsize)
    return figure, figure.subplots()


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _placeholder(title: str, message: str, figsize: Tuple[float, float]) -> Figure:
    """A figure that says why there is nothing to draw (used for empty/missing data)."""
    figure, ax = _new_axes(figsize)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes,
            fontsize=11, color="#555555")
    ax.set_axis_off()
    figure.tight_layout()
    return figure


def _top_limit(values: List[float], headroom: float = 1.18) -> float:
    """Upper y-limit: a little above the largest value, or 1.0 when everything is 0."""
    largest = max(values) if values else 0.0
    return largest * headroom if largest > 0 else 1.0


def _format(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return f"{value:.0f}"
    return f"{value:.3f}" if abs(value) < 0.1 else f"{value:.2f}"


def _improvement_text(row: MetricComparison) -> str:
    """One neutral sentence taken from the numbers metrics.py already computed."""
    percent = row.improvement_percent
    if row.winner is Winner.TIE:
        return "No difference between the controllers"
    if percent is None:
        return "Percentage difference undefined (fixed-time value is 0)"
    if percent > 0:
        return f"Adaptive is {percent:.1f}% better"
    if percent < 0:
        return f"Adaptive is {abs(percent):.1f}% worse"
    return "No difference between the controllers"


# --------------------------------------------------------------------------- #
# 1-6. Controller comparison bar charts (two bars each)
# --------------------------------------------------------------------------- #
def _plot_metric_bars(
    comparison: MetricsComparison,
    metric_name: str,
    title: str,
    xlabel: str = "Controller",
) -> Figure:
    _require_comparison(comparison)
    row = comparison.row(metric_name)   # ValueError for an unknown metric
    values = [_finite("fixed_time value", row.fixed_time), _finite("adaptive value", row.adaptive)]
    names = [str(comparison.fixed_time.controller_name), str(comparison.adaptive.controller_name)]

    figure, ax = _new_axes(FIGSIZE_BARS)
    positions = [0, 1]
    bars = ax.bar(positions, values, width=0.55, color=[FIXED_COLOR, ADAPTIVE_COLOR])
    ax.bar_label(bars, labels=[_format(v) for v in values], padding=3, fontsize=10)
    ax.set_xticks(positions)
    ax.set_xticklabels(names)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, _top_limit(values))      # bars always start at zero

    unit = _UNIT_NAMES.get(row.unit, row.unit)
    better = "higher is better" if row.higher_is_better else "lower is better"
    _style(ax, f"{title}\n{_improvement_text(row)}", xlabel, f"{unit} ({better})")
    figure.tight_layout()
    return figure


def plot_average_waiting_time(comparison: MetricsComparison) -> Figure:
    """Average waiting time of all vehicles, Fixed-Time vs Adaptive (2 bars)."""
    return _plot_metric_bars(comparison, "average_waiting_time", "Average Vehicle Waiting Time")


def plot_max_waiting_time(comparison: MetricsComparison) -> Figure:
    """Longest single waiting time, Fixed-Time vs Adaptive (2 bars)."""
    return _plot_metric_bars(comparison, "max_waiting_time", "Maximum Vehicle Waiting Time")


def plot_average_queue_length(comparison: MetricsComparison) -> Figure:
    """Average queue length per lane, Fixed-Time vs Adaptive (2 bars)."""
    return _plot_metric_bars(comparison, "average_queue_length", "Average Queue Length (per lane)")


def plot_max_queue_length(comparison: MetricsComparison) -> Figure:
    """Longest queue of any single lane, Fixed-Time vs Adaptive (2 bars)."""
    return _plot_metric_bars(comparison, "max_queue_length", "Maximum Queue Length (single lane)")


def plot_throughput(comparison: MetricsComparison) -> Figure:
    """Vehicles served per minute, Fixed-Time vs Adaptive (2 bars)."""
    return _plot_metric_bars(comparison, "throughput_per_minute", "Throughput")


def plot_congestion_index(comparison: MetricsComparison) -> Figure:
    """Share of lane-seconds spent congested, Fixed-Time vs Adaptive (2 bars)."""
    _require_comparison(comparison)
    threshold = comparison.fixed_time.congestion_threshold
    return _plot_metric_bars(
        comparison,
        "congestion_index",
        "Congestion Index",
        xlabel=f"Controller (a lane counts as congested at {threshold} or more vehicles)",
    )


# --------------------------------------------------------------------------- #
# 7. Direction-wise average queue (grouped bars)
# --------------------------------------------------------------------------- #
def plot_direction_average_queue(comparison: MetricsComparison) -> Figure:
    """Average queue length of each direction: 4 groups, each with a Fixed-Time and an Adaptive bar."""
    _require_comparison(comparison)
    fixed_values = _direction_values("fixed_time.average_queue_by_direction",
                                     comparison.fixed_time.average_queue_by_direction)
    adaptive_values = _direction_values("adaptive.average_queue_by_direction",
                                        comparison.adaptive.average_queue_by_direction)

    figure, ax = _new_axes(FIGSIZE_GROUPED)
    width = 0.38
    centres = list(range(len(_DIRECTIONS)))
    fixed_bars = ax.bar([c - width / 2 for c in centres], fixed_values, width,
                        color=FIXED_COLOR, label=str(comparison.fixed_time.controller_name))
    adaptive_bars = ax.bar([c + width / 2 for c in centres], adaptive_values, width,
                           color=ADAPTIVE_COLOR, label=str(comparison.adaptive.controller_name))
    ax.bar_label(fixed_bars, labels=[_format(v) for v in fixed_values], padding=2, fontsize=9)
    ax.bar_label(adaptive_bars, labels=[_format(v) for v in adaptive_values], padding=2, fontsize=9)

    ax.set_xticks(centres)
    ax.set_xticklabels([d.value for d in _DIRECTIONS])
    ax.set_ylim(0, _top_limit(fixed_values + adaptive_values))
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    _style(ax, "Average Queue Length by Direction", "Direction",
           "vehicles waiting (average over time, lower is better)")
    figure.tight_layout()
    return figure


# --------------------------------------------------------------------------- #
# Time-series helpers (private)
# --------------------------------------------------------------------------- #
def _collect(result: SimulationResult, getter: Callable[[Any], Any]):
    """Read one value per history record.

    Returns ``(times, values)`` (both empty for an empty history), or ``None``
    if a record lacks a field the getter needs (the data is "not available").
    """
    times: List[int] = []
    values: List[Any] = []
    for index, record in enumerate(result.history):
        try:
            value = getter(record)
        except (AttributeError, KeyError, TypeError):
            return None
        times.append(getattr(record, "time", index))
        values.append(value)
    return times, values


# --------------------------------------------------------------------------- #
# 8. Queue length over time
# --------------------------------------------------------------------------- #
def plot_queue_over_time(
    fixed_time: SimulationResult,
    adaptive: SimulationResult,
    direction: Optional[Direction] = None,
) -> Figure:
    """Queue length second by second, one line per controller.

    ``direction=None``: all four lanes added together.
    ``direction=Direction.NORTH`` (etc.): only that lane.
    """
    _require_result("fixed_time", fixed_time)
    _require_result("adaptive", adaptive)
    if direction is not None and not isinstance(direction, Direction):
        raise ValueError(f"direction must be a Direction or None, got {direction!r}")

    if direction is None:
        what, getter = "all lanes", (lambda record: sum(record.queue_lengths.values()))
        ylabel = "vehicles waiting (all lanes)"
    else:
        what, getter = f"{direction.value} lane", (lambda record: record.queue_lengths[direction])
        ylabel = f"vehicles waiting ({direction.value} lane)"
    title = f"Queue Length Over Time ({what})"

    series = []
    for result, color in ((fixed_time, FIXED_COLOR), (adaptive, ADAPTIVE_COLOR)):
        data = _collect(result, getter)
        if data is None:
            return _placeholder(title, _NO_QUEUE_DATA, FIGSIZE_TIME)
        series.append((str(result.controller_name), color, data))
    if all(not times for _, _, (times, _) in series):
        return _placeholder(title, _NO_DATA, FIGSIZE_TIME)

    figure, ax = _new_axes(FIGSIZE_TIME)
    largest = 0.0
    for name, color, (times, values) in series:
        if times:
            ax.plot(times, values, color=color, label=name, linewidth=1.6, drawstyle="steps-post")
            largest = max(largest, max(values))
    ax.set_ylim(0, largest * 1.1 if largest > 0 else 1.0)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    _style(ax, title, "Simulation time (s)", ylabel)
    figure.tight_layout()
    return figure


def plot_direction_queues_over_time(result: SimulationResult) -> Figure:
    """Queue length second by second for ONE controller, one line per direction."""
    _require_result("result", result)
    title = f"Queue Length by Direction Over Time - {result.controller_name}"
    data = _collect(result, lambda record: [record.queue_lengths[d] for d in _DIRECTIONS])
    if data is None:
        return _placeholder(title, _NO_QUEUE_DATA, FIGSIZE_TIME)
    times, rows = data
    if not times:
        return _placeholder(title, _NO_DATA, FIGSIZE_TIME)

    figure, ax = _new_axes(FIGSIZE_TIME)
    largest = 0.0
    for index, direction in enumerate(_DIRECTIONS):
        values = [row[index] for row in rows]
        ax.plot(times, values, color=DIRECTION_COLORS[direction], label=direction.value,
                linewidth=1.5, drawstyle="steps-post")
        largest = max(largest, max(values))
    ax.set_ylim(0, largest * 1.1 if largest > 0 else 1.0)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    _style(ax, title, "Simulation time (s)", "vehicles waiting")
    figure.tight_layout()
    return figure


# --------------------------------------------------------------------------- #
# 9. Signal states over time
# --------------------------------------------------------------------------- #
def plot_signal_states(
    result: SimulationResult,
    start: int = 0,
    end: Optional[int] = None,
) -> Figure:
    """Timeline of the signal colour of each direction (one row per direction).

    ``start`` / ``end`` select a window in seconds (``end`` is exclusive,
    ``None`` = until the last simulated second). A whole hour is hard to read,
    so a window of a few minutes is usually the useful view.
    """
    _require_result("result", result)
    _check_window(start, end)
    title = f"Signal States Over Time - {result.controller_name}"

    data = _collect(result, lambda record: dict(record.signals))
    if data is None:
        return _placeholder(title, _NO_SIGNAL_DATA, FIGSIZE_SIGNALS)
    times, signal_rows = data
    selected = [
        (t, signals) for t, signals in zip(times, signal_rows)
        if t >= start and (end is None or t < end)
    ]
    if not selected:
        return _placeholder(title, _NO_DATA, FIGSIZE_SIGNALS)

    # Run-length encoding: consecutive seconds with the same colour become ONE segment.
    segments: Dict[Direction, Dict[SignalState, List[Tuple[int, int]]]] = {}
    for direction in _DIRECTIONS:
        per_state: Dict[SignalState, List[Tuple[int, int]]] = {state: [] for state in SignalState}
        current_state, segment_start, previous_time = None, None, None
        for t, signals in selected:
            state = signals.get(direction) if isinstance(signals, Mapping) else None
            if not isinstance(state, SignalState):
                raise ValueError(f"signal of {direction.name} at second {t} must be a SignalState, got {state!r}")
            continues = state is current_state and previous_time is not None and t == previous_time + 1
            if not continues:
                if current_state is not None:
                    per_state[current_state].append((segment_start, previous_time + 1 - segment_start))
                current_state, segment_start = state, t
            previous_time = t
        per_state[current_state].append((segment_start, previous_time + 1 - segment_start))
        segments[direction] = per_state

    figure, ax = _new_axes(FIGSIZE_SIGNALS)
    rows = len(_DIRECTIONS)
    for index, direction in enumerate(_DIRECTIONS):
        y = rows - 1 - index                       # North on top
        for state in (SignalState.GREEN, SignalState.YELLOW, SignalState.RED):
            ranges = segments[direction][state]
            if ranges:
                ax.broken_barh(ranges, (y - 0.4, 0.8), facecolors=SIGNAL_COLORS[state], edgecolor="none")

    ax.set_yticks(list(range(rows)))
    ax.set_yticklabels([d.value for d in reversed(_DIRECTIONS)])
    ax.set_ylim(-0.6, rows - 0.4)
    ax.set_xlim(selected[0][0], selected[-1][0] + 1)
    handles = [Patch(facecolor=SIGNAL_COLORS[s], label=SIGNAL_LABELS[s])
               for s in (SignalState.GREEN, SignalState.YELLOW, SignalState.RED)]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    _style(ax, title, "Simulation time (s)", "Direction")
    ax.grid(axis="y", visible=False)
    figure.tight_layout()
    return figure


# --------------------------------------------------------------------------- #
# 10. Arrivals and departures over time
# --------------------------------------------------------------------------- #
def plot_arrivals_departures(result: SimulationResult) -> Figure:
    """Cumulative vehicles arrived and departed over time for ONE controller.

    The vertical gap between the two lines is the number of vehicles still waiting.
    """
    _require_result("result", result)
    title = f"Cumulative Arrivals and Departures - {result.controller_name}"
    data = _collect(result, lambda record: (record.total_arrived, record.total_departed))
    if data is None:
        return _placeholder(title, _NO_FLOW_DATA, FIGSIZE_TIME)
    times, pairs = data
    if not times:
        return _placeholder(title, _NO_DATA, FIGSIZE_TIME)

    arrived = [pair[0] for pair in pairs]
    departed = [pair[1] for pair in pairs]
    figure, ax = _new_axes(FIGSIZE_TIME)
    ax.plot(times, arrived, color="#444444", linewidth=1.8, label="Cumulative arrivals")
    departure_color = ADAPTIVE_COLOR if result.controller_type is ControllerType.ADAPTIVE else FIXED_COLOR
    ax.plot(times, departed, color=departure_color, linewidth=1.8, label="Cumulative departures")
    top = max(arrived + departed)
    ax.set_ylim(0, top * 1.08 if top > 0 else 1.0)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    _style(ax, title, "Simulation time (s)", "vehicles (gap between lines = vehicles waiting)")
    figure.tight_layout()
    return figure


# --------------------------------------------------------------------------- #
# Convenience for a future dashboard
# --------------------------------------------------------------------------- #
def build_all_figures(
    comparison: ComparisonResult,
    metrics_comparison: MetricsComparison,
    signal_window: Tuple[int, Optional[int]] = (0, 300),
) -> Dict[str, Figure]:
    """Draw every chart from data that already exists; returns ``{name: Figure}``.

    ``comparison`` is what ``run_comparison`` returned, ``metrics_comparison``
    is what ``evaluate_comparison`` returned for it. This function only draws.
    """
    if not isinstance(comparison, ComparisonResult):
        raise ValueError(f"comparison must be a ComparisonResult, got {comparison!r}")
    _require_comparison(metrics_comparison)
    if not isinstance(signal_window, tuple) or len(signal_window) != 2:
        raise ValueError("signal_window must be a (start, end) tuple")
    start, end = signal_window
    fixed, adaptive = comparison.fixed_time, comparison.adaptive

    return {
        "average_waiting_time": plot_average_waiting_time(metrics_comparison),
        "max_waiting_time": plot_max_waiting_time(metrics_comparison),
        "average_queue_length": plot_average_queue_length(metrics_comparison),
        "max_queue_length": plot_max_queue_length(metrics_comparison),
        "throughput": plot_throughput(metrics_comparison),
        "congestion_index": plot_congestion_index(metrics_comparison),
        "direction_average_queue": plot_direction_average_queue(metrics_comparison),
        "queue_over_time": plot_queue_over_time(fixed, adaptive),
        "direction_queues_fixed_time": plot_direction_queues_over_time(fixed),
        "direction_queues_adaptive": plot_direction_queues_over_time(adaptive),
        "signal_states_fixed_time": plot_signal_states(fixed, start, end),
        "signal_states_adaptive": plot_signal_states(adaptive, start, end),
        "arrivals_departures_fixed_time": plot_arrivals_departures(fixed),
        "arrivals_departures_adaptive": plot_arrivals_departures(adaptive),
    }


def figure_to_png_bytes(figure: Figure, dpi: int = 150) -> bytes:
    """Render a figure to PNG bytes (e.g. for a Streamlit download button)."""
    if not isinstance(figure, Figure):
        raise ValueError(f"figure must be a matplotlib Figure, got {figure!r}")
    if isinstance(dpi, bool) or not isinstance(dpi, Integral) or dpi < 1:
        raise ValueError(f"dpi must be a whole number >= 1, got {dpi!r}")
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=int(dpi), bbox_inches="tight")
    return buffer.getvalue()
