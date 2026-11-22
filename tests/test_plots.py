"""Tests for plots.py.

We test STRUCTURE and DATA (how many bars/lines, which numbers they hold, titles,
labels, empty and invalid input), never pixels.

Most tests use HAND-BUILT simulation results so every expected number can be
checked on paper. A few use the real simulator, only to prove the plots work
with real data.
"""

import math
import unittest
from dataclasses import replace
from types import SimpleNamespace
from typing import List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")   # non-interactive backend: no windows are ever opened

import matplotlib.pyplot as plt                      # noqa: E402  (only used to check for leaked figures)
from matplotlib.colors import to_hex                 # noqa: E402
from matplotlib.figure import Figure                 # noqa: E402

import plots                                         # noqa: E402
from config import Direction, SimulationConfig       # noqa: E402
from controllers import ControllerDecision           # noqa: E402
from coordinator import Action                       # noqa: E402
from junction import Phase, SignalState, Stage       # noqa: E402
from metrics import (                                # noqa: E402
    SimulationMetrics,
    calculate_metrics,
    compare_metrics,
    compare_results,
    evaluate_comparison,
)
from simulator import (                              # noqa: E402
    ControllerType,
    SimulationResult,
    StepRecord,
    VehicleRecord,
    run_comparison,
)

N, S, E, W = Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST
FIXED, ADAPTIVE = ControllerType.FIXED_TIME, ControllerType.ADAPTIVE
NAMES = {FIXED: "Fixed-Time", ADAPTIVE: "Adaptive Multi-Agent"}
GREEN, YELLOW, RED = SignalState.GREEN, SignalState.YELLOW, SignalState.RED


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def cfg(duration: int = 100, **changes) -> SimulationConfig:
    return replace(SimulationConfig(), duration=duration, **changes)


def all_red(t: int):
    return {d: RED for d in Direction}


def make_result(vehicles, steps, duration=None, controller_type=FIXED, signals_for=all_red) -> SimulationResult:
    """Hand-built, consistent SimulationResult (history derived from the vehicle list)."""
    duration = max(steps, 1) if duration is None else duration
    records = tuple(VehicleRecord(i, d, a, dep) for i, d, a, dep in vehicles)
    history = []
    queue = {d: 0 for d in Direction}
    total_arrived = total_departed = 0
    for t in range(steps):
        arrived = {d: sum(1 for v in vehicles if v[1] is d and v[2] == t) for d in Direction}
        departed = {d: sum(1 for v in vehicles if v[1] is d and v[3] == t) for d in Direction}
        for d in Direction:
            queue[d] += arrived[d] - departed[d]
        total_arrived += sum(arrived.values())
        total_departed += sum(departed.values())
        history.append(StepRecord(
            time=t, stage=Stage.GREEN, active_phase=Phase.NORTH_SOUTH, signals=signals_for(t),
            queue_lengths=dict(queue), arrived=arrived, departed=departed,
            total_arrived=total_arrived, total_departed=total_departed,
            decision=ControllerDecision(Action.NONE, None, 0),
        ))
    return SimulationResult(cfg(duration), controller_type, NAMES[controller_type], tuple(history), records)


# Fixed-time scenario (same numbers as the metrics tests, calculated by hand there):
#   queue per second  N: 1,2,1,1,0,0,0,0,0,0   total: 1,2,2,3,2,2,2,2,2,2
#   cumulative arrived 1,2,3,5,5,6,6,6,6,6 ; departed 0,0,1,2,3,4,4,4,4,4
FIXED_VEHICLES = [(1, N, 0, 2), (2, N, 1, 4), (3, S, 2, 3), (4, E, 3, None), (5, E, 3, None), (6, W, 5, 5)]
# The same vehicles served faster by the "adaptive" run: total queue 1,1,1,2,1,0,0,0,0,0
ADAPTIVE_VEHICLES = [(1, N, 0, 1), (2, N, 1, 2), (3, S, 2, 3), (4, E, 3, 4), (5, E, 3, 5), (6, W, 5, 5)]


def fixed_result() -> SimulationResult:
    return make_result(FIXED_VEHICLES, 10, controller_type=FIXED)


def adaptive_result() -> SimulationResult:
    return make_result(ADAPTIVE_VEHICLES, 10, controller_type=ADAPTIVE)


def empty_result(controller_type=FIXED) -> SimulationResult:
    return make_result([], steps=0, duration=10, controller_type=controller_type)


def metrics_comparison(fixed=None, adaptive=None):
    return compare_results(fixed or fixed_result(), adaptive or adaptive_result(), congestion_threshold=2)


def signal_cycle(t: int):
    """t 0-2: N+S green; 3: N+S yellow; 4: all red; 5-7: E+W green."""
    if t <= 2:
        return {N: GREEN, S: GREEN, E: RED, W: RED}
    if t == 3:
        return {N: YELLOW, S: YELLOW, E: RED, W: RED}
    if t == 4:
        return all_red(t)
    return {N: RED, S: RED, E: GREEN, W: GREEN}


def without_fields(result: SimulationResult, *missing: str) -> SimulationResult:
    """Copy whose history records lack some fields (to test 'optional fields missing')."""
    fields = ("time", "signals", "queue_lengths", "arrived", "departed", "total_arrived", "total_departed")
    history = tuple(
        SimpleNamespace(**{f: getattr(r, f) for f in fields if f not in missing}) for r in result.history
    )
    return replace(result, history=history)


def bar_heights(ax) -> List[float]:
    return [p.get_height() for p in ax.patches]


def xtick_texts(ax) -> List[str]:
    return [t.get_text() for t in ax.get_xticklabels()]


def legend_texts(ax) -> List[str]:
    legend = ax.get_legend()
    return [t.get_text() for t in legend.get_texts()] if legend else []


def message_of(figure: Figure) -> str:
    return " ".join(t.get_text() for t in figure.axes[0].texts)


BAR_FUNCTIONS = {
    "average_waiting_time": plots.plot_average_waiting_time,
    "max_waiting_time": plots.plot_max_waiting_time,
    "average_queue_length": plots.plot_average_queue_length,
    "max_queue_length": plots.plot_max_queue_length,
    "throughput_per_minute": plots.plot_throughput,
    "congestion_index": plots.plot_congestion_index,
}


# --------------------------------------------------------------------------- #
# 1-6. Controller comparison bar charts
# --------------------------------------------------------------------------- #
class TestComparisonBarCharts(unittest.TestCase):
    def setUp(self):
        self.comparison = metrics_comparison()

    def test_each_function_returns_a_figure_with_one_axes_and_two_bars(self):
        for name, function in BAR_FUNCTIONS.items():
            with self.subTest(metric=name):
                figure = function(self.comparison)
                self.assertIsInstance(figure, Figure)
                self.assertEqual(len(figure.axes), 1)
                self.assertEqual(len(figure.axes[0].patches), 2)

    def test_bar_heights_are_exactly_the_metric_values(self):
        for name, function in BAR_FUNCTIONS.items():
            with self.subTest(metric=name):
                row = self.comparison.row(name)
                heights = bar_heights(function(self.comparison).axes[0])
                self.assertAlmostEqual(heights[0], row.fixed_time)
                self.assertAlmostEqual(heights[1], row.adaptive)

    def test_known_values(self):
        """Hand-calculated: average wait 20/6 vs 1.0; throughput 24 vs 36 vehicles/min."""
        wait = bar_heights(plots.plot_average_waiting_time(self.comparison).axes[0])
        self.assertAlmostEqual(wait[0], 20 / 6)
        self.assertAlmostEqual(wait[1], 1.0)
        throughput = bar_heights(plots.plot_throughput(self.comparison).axes[0])
        self.assertAlmostEqual(throughput[0], 24.0)
        self.assertAlmostEqual(throughput[1], 36.0)

    def test_titles_labels_and_controller_names(self):
        expected_titles = {
            "average_waiting_time": "Average Vehicle Waiting Time",
            "max_waiting_time": "Maximum Vehicle Waiting Time",
            "average_queue_length": "Average Queue Length",
            "max_queue_length": "Maximum Queue Length",
            "throughput_per_minute": "Throughput",
            "congestion_index": "Congestion Index",
        }
        for name, function in BAR_FUNCTIONS.items():
            with self.subTest(metric=name):
                ax = function(self.comparison).axes[0]
                self.assertIn(expected_titles[name], ax.get_title())
                self.assertTrue(ax.get_xlabel().startswith("Controller"))
                self.assertTrue(ax.get_ylabel())
                self.assertEqual(xtick_texts(ax), ["Fixed-Time", "Adaptive Multi-Agent"])

    def test_axis_label_states_which_direction_is_better(self):
        self.assertIn("lower is better", plots.plot_average_waiting_time(self.comparison).axes[0].get_ylabel())
        self.assertIn("higher is better", plots.plot_throughput(self.comparison).axes[0].get_ylabel())

    def test_congestion_chart_mentions_the_threshold(self):
        ax = plots.plot_congestion_index(self.comparison).axes[0]
        self.assertIn("2 or more vehicles", ax.get_xlabel())

    def test_bars_start_at_zero_and_leave_headroom(self):
        for name, function in BAR_FUNCTIONS.items():
            with self.subTest(metric=name):
                ax = function(self.comparison).axes[0]
                low, high = ax.get_ylim()
                self.assertEqual(low, 0)
                self.assertGreater(high, max(bar_heights(ax)))

    def test_consistent_colours(self):
        for name, function in BAR_FUNCTIONS.items():
            with self.subTest(metric=name):
                patches = function(self.comparison).axes[0].patches
                self.assertEqual(to_hex(patches[0].get_facecolor()), to_hex(plots.FIXED_COLOR))
                self.assertEqual(to_hex(patches[1].get_facecolor()), to_hex(plots.ADAPTIVE_COLOR))

    def test_each_bar_has_a_value_label(self):
        ax = plots.plot_average_waiting_time(self.comparison).axes[0]
        self.assertEqual(sorted(t.get_text() for t in ax.texts), ["1", "3.33"])

    def test_improvement_text_when_adaptive_is_better(self):
        title = plots.plot_average_waiting_time(self.comparison).axes[0].get_title()
        self.assertIn("Adaptive is", title)
        self.assertIn("better", title)
        self.assertIn("70.0%", title)   # (3.333 - 1.0) / 3.333

    def test_improvement_text_when_adaptive_is_worse(self):
        """The chart must never flatter the adaptive controller."""
        # Swap the roles: serve slowly in the "adaptive" run, quickly in the fixed-time run.
        fast_fixed = make_result(ADAPTIVE_VEHICLES, 10, controller_type=FIXED)
        slow_adaptive = make_result(FIXED_VEHICLES, 10, controller_type=ADAPTIVE)
        worse = compare_results(fast_fixed, slow_adaptive, congestion_threshold=2)
        title = plots.plot_average_waiting_time(worse).axes[0].get_title()
        self.assertIn("worse", title)
        self.assertNotIn("better", title)

    def test_identical_controllers_show_no_difference(self):
        same = compare_results(
            make_result(FIXED_VEHICLES, 10, controller_type=FIXED),
            make_result(FIXED_VEHICLES, 10, controller_type=ADAPTIVE),
        )
        title = plots.plot_average_waiting_time(same).axes[0].get_title()
        self.assertIn("No difference", title)

    def test_all_zero_values(self):
        empty = compare_results(make_result([], steps=5, controller_type=FIXED),
                                make_result([], steps=5, controller_type=ADAPTIVE))
        for name, function in BAR_FUNCTIONS.items():
            with self.subTest(metric=name):
                ax = function(empty).axes[0]
                self.assertEqual(bar_heights(ax), [0.0, 0.0])
                self.assertEqual(ax.get_ylim(), (0, 1.0))      # sensible axis, not a degenerate 0..0
                self.assertIn("No difference", ax.get_title())

    def test_zero_baseline_says_percentage_is_undefined(self):
        fixed_metrics = replace(self.comparison.fixed_time, throughput_per_minute=0.0)
        comparison = compare_metrics(fixed_metrics, self.comparison.adaptive)
        title = plots.plot_throughput(comparison).axes[0].get_title()
        self.assertIn("undefined", title)

    def test_no_pyplot_figure_is_created(self):
        before = plt.get_fignums()
        for function in BAR_FUNCTIONS.values():
            function(self.comparison)
        self.assertEqual(plt.get_fignums(), before)

    def test_invalid_inputs(self):
        for name, function in BAR_FUNCTIONS.items():
            for bad in (None, "comparison", 42, self.comparison.fixed_time, [self.comparison]):
                with self.subTest(metric=name, bad=bad):
                    with self.assertRaises(ValueError):
                        function(bad)

    def test_non_finite_values_are_rejected(self):
        rows = tuple(replace(r, fixed_time=float("nan")) if r.name == "average_waiting_time" else r
                     for r in self.comparison.rows)
        with self.assertRaises(ValueError):
            plots.plot_average_waiting_time(replace(self.comparison, rows=rows))

    def test_missing_metric_row_is_rejected(self):
        rows = tuple(r for r in self.comparison.rows if r.name != "throughput_per_minute")
        with self.assertRaises(ValueError):
            plots.plot_throughput(replace(self.comparison, rows=rows))


# --------------------------------------------------------------------------- #
# 7. Direction-wise grouped bars
# --------------------------------------------------------------------------- #
class TestDirectionAverageQueue(unittest.TestCase):
    def setUp(self):
        self.comparison = metrics_comparison()
        self.figure = plots.plot_direction_average_queue(self.comparison)
        self.ax = self.figure.axes[0]

    def test_eight_bars_in_four_groups(self):
        self.assertIsInstance(self.figure, Figure)
        self.assertEqual(len(self.figure.axes), 1)
        self.assertEqual(len(self.ax.patches), 8)
        self.assertEqual(len(self.ax.containers), 2)

    def test_bar_heights_match_the_metrics_in_north_south_east_west_order(self):
        fixed_expected = [self.comparison.fixed_time.average_queue_by_direction[d] for d in Direction]
        adaptive_expected = [self.comparison.adaptive.average_queue_by_direction[d] for d in Direction]
        heights = bar_heights(self.ax)
        for got, want in zip(heights[:4], fixed_expected):
            self.assertAlmostEqual(got, want)
        for got, want in zip(heights[4:], adaptive_expected):
            self.assertAlmostEqual(got, want)

    def test_known_fixed_time_values(self):
        """Hand-calculated in the metrics tests: N 0.5, S 0.1, E 1.4, W 0.0."""
        for got, want in zip(bar_heights(self.ax)[:4], [0.5, 0.1, 1.4, 0.0]):
            self.assertAlmostEqual(got, want)

    def test_labels_legend_and_title(self):
        self.assertEqual(xtick_texts(self.ax), ["North", "South", "East", "West"])
        self.assertEqual(legend_texts(self.ax), ["Fixed-Time", "Adaptive Multi-Agent"])
        self.assertIn("Direction", self.ax.get_title())
        self.assertEqual(self.ax.get_xlabel(), "Direction")
        self.assertTrue(self.ax.get_ylabel())

    def test_colours_and_zero_baseline(self):
        patches = self.ax.patches
        self.assertTrue(all(to_hex(p.get_facecolor()) == to_hex(plots.FIXED_COLOR) for p in patches[:4]))
        self.assertTrue(all(to_hex(p.get_facecolor()) == to_hex(plots.ADAPTIVE_COLOR) for p in patches[4:]))
        self.assertEqual(self.ax.get_ylim()[0], 0)
        self.assertGreater(self.ax.get_ylim()[1], max(bar_heights(self.ax)))

    def test_all_zero_queues(self):
        empty = compare_results(make_result([], steps=5, controller_type=FIXED),
                                make_result([], steps=5, controller_type=ADAPTIVE))
        ax = plots.plot_direction_average_queue(empty).axes[0]
        self.assertEqual(bar_heights(ax), [0.0] * 8)
        self.assertEqual(ax.get_ylim(), (0, 1.0))

    def test_invalid_input(self):
        for bad in (None, "x", self.comparison.fixed_time):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    plots.plot_direction_average_queue(bad)

    def test_missing_direction_is_rejected(self):
        partial = {d: v for d, v in self.comparison.fixed_time.average_queue_by_direction.items() if d is not W}
        bad = replace(self.comparison, fixed_time=replace(self.comparison.fixed_time, average_queue_by_direction=partial))
        with self.assertRaises(ValueError):
            plots.plot_direction_average_queue(bad)

    def test_negative_or_non_numeric_values_are_rejected(self):
        for bad_value in (-1.0, float("nan"), "3", None):
            with self.subTest(bad_value=bad_value):
                values = dict(self.comparison.adaptive.average_queue_by_direction)
                values[N] = bad_value
                bad = replace(self.comparison,
                              adaptive=replace(self.comparison.adaptive, average_queue_by_direction=values))
                with self.assertRaises(ValueError):
                    plots.plot_direction_average_queue(bad)


# --------------------------------------------------------------------------- #
# 8. Queue length over time
# --------------------------------------------------------------------------- #
class TestQueueOverTime(unittest.TestCase):
    def test_total_queue_has_one_line_per_controller(self):
        figure = plots.plot_queue_over_time(fixed_result(), adaptive_result())
        ax = figure.axes[0]
        self.assertEqual(len(ax.lines), 2)
        self.assertEqual(list(ax.lines[0].get_xdata()), list(range(10)))
        self.assertEqual(list(ax.lines[0].get_ydata()), [1, 2, 2, 3, 2, 2, 2, 2, 2, 2])
        self.assertEqual(list(ax.lines[1].get_ydata()), [1, 1, 1, 2, 1, 0, 0, 0, 0, 0])

    def test_labels_legend_colours(self):
        ax = plots.plot_queue_over_time(fixed_result(), adaptive_result()).axes[0]
        self.assertEqual(legend_texts(ax), ["Fixed-Time", "Adaptive Multi-Agent"])
        self.assertIn("Queue Length Over Time", ax.get_title())
        self.assertIn("all lanes", ax.get_title())
        self.assertIn("time", ax.get_xlabel().lower())
        self.assertIn("vehicles", ax.get_ylabel())
        self.assertEqual(to_hex(ax.lines[0].get_color()), to_hex(plots.FIXED_COLOR))
        self.assertEqual(to_hex(ax.lines[1].get_color()), to_hex(plots.ADAPTIVE_COLOR))
        self.assertEqual(ax.get_ylim()[0], 0)
        self.assertGreater(ax.get_ylim()[1], 3)

    def test_single_direction(self):
        ax = plots.plot_queue_over_time(fixed_result(), adaptive_result(), direction=N).axes[0]
        self.assertEqual(len(ax.lines), 2)
        self.assertEqual(list(ax.lines[0].get_ydata()), [1, 2, 1, 1, 0, 0, 0, 0, 0, 0])
        self.assertIn("North", ax.get_title())
        self.assertIn("North", ax.get_ylabel())

    def test_every_direction_can_be_plotted(self):
        for direction in Direction:
            with self.subTest(direction=direction):
                ax = plots.plot_queue_over_time(fixed_result(), adaptive_result(), direction).axes[0]
                self.assertEqual(len(ax.lines), 2)
                self.assertIn(direction.value, ax.get_title())

    def test_all_zero_queues_still_get_a_sensible_axis(self):
        quiet = make_result([], steps=5)
        ax = plots.plot_queue_over_time(quiet, make_result([], steps=5, controller_type=ADAPTIVE)).axes[0]
        self.assertEqual(list(ax.lines[0].get_ydata()), [0] * 5)
        self.assertEqual(ax.get_ylim(), (0, 1.0))

    def test_empty_history_gives_a_message_not_a_crash(self):
        figure = plots.plot_queue_over_time(empty_result(FIXED), empty_result(ADAPTIVE))
        ax = figure.axes[0]
        self.assertEqual(len(ax.lines), 0)
        self.assertIn("No simulation data", message_of(figure))

    def test_one_empty_history_plots_only_the_other_line(self):
        ax = plots.plot_queue_over_time(fixed_result(), empty_result(ADAPTIVE)).axes[0]
        self.assertEqual(len(ax.lines), 1)

    def test_runs_of_different_length_are_plotted_as_they_are(self):
        short = make_result(FIXED_VEHICLES[:2], steps=5, controller_type=ADAPTIVE)
        ax = plots.plot_queue_over_time(fixed_result(), short).axes[0]
        self.assertEqual(len(ax.lines[0].get_xdata()), 10)
        self.assertEqual(len(ax.lines[1].get_xdata()), 5)

    def test_missing_queue_field_gives_a_message(self):
        figure = plots.plot_queue_over_time(without_fields(fixed_result(), "queue_lengths"), adaptive_result())
        self.assertEqual(len(figure.axes[0].lines), 0)
        self.assertIn("not available", message_of(figure))

    def test_missing_optional_fields_do_not_matter(self):
        """Queue plots need only time + queue_lengths; signals/arrivals may be absent."""
        stripped = without_fields(fixed_result(), "signals", "arrived", "departed", "total_arrived", "total_departed")
        ax = plots.plot_queue_over_time(stripped, adaptive_result()).axes[0]
        self.assertEqual(len(ax.lines), 2)

    def test_invalid_inputs(self):
        for bad in (None, "result", 5):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    plots.plot_queue_over_time(bad, adaptive_result())
                with self.assertRaises(ValueError):
                    plots.plot_queue_over_time(fixed_result(), bad)
        for bad_direction in ("North", 1, Phase.NORTH_SOUTH):
            with self.subTest(bad_direction=bad_direction):
                with self.assertRaises(ValueError):
                    plots.plot_queue_over_time(fixed_result(), adaptive_result(), direction=bad_direction)

    def test_no_pyplot_figure_is_created(self):
        before = plt.get_fignums()
        plots.plot_queue_over_time(fixed_result(), adaptive_result())
        self.assertEqual(plt.get_fignums(), before)


class TestDirectionQueuesOverTime(unittest.TestCase):
    def test_four_lines_with_expected_data(self):
        figure = plots.plot_direction_queues_over_time(fixed_result())
        ax = figure.axes[0]
        self.assertEqual(len(ax.lines), 4)
        self.assertEqual(legend_texts(ax), ["North", "South", "East", "West"])
        self.assertEqual(list(ax.lines[0].get_ydata()), [1, 2, 1, 1, 0, 0, 0, 0, 0, 0])      # North
        self.assertEqual(list(ax.lines[2].get_ydata()), [0, 0, 0, 2, 2, 2, 2, 2, 2, 2])      # East
        self.assertIn("Fixed-Time", ax.get_title())
        self.assertEqual(ax.get_ylim()[0], 0)
        for line, direction in zip(ax.lines, Direction):
            self.assertEqual(to_hex(line.get_color()), to_hex(plots.DIRECTION_COLORS[direction]))

    def test_empty_history(self):
        figure = plots.plot_direction_queues_over_time(empty_result())
        self.assertEqual(len(figure.axes[0].lines), 0)
        self.assertIn("No simulation data", message_of(figure))

    def test_missing_queue_field(self):
        figure = plots.plot_direction_queues_over_time(without_fields(fixed_result(), "queue_lengths"))
        self.assertIn("not available", message_of(figure))

    def test_invalid_input(self):
        for bad in (None, "x", fixed_result().history):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    plots.plot_direction_queues_over_time(bad)


# --------------------------------------------------------------------------- #
# 9. Signal states over time
# --------------------------------------------------------------------------- #
class TestSignalStates(unittest.TestCase):
    def setUp(self):
        self.result = make_result([], steps=8, signals_for=signal_cycle)

    def test_whole_timeline(self):
        figure = plots.plot_signal_states(self.result)
        ax = figure.axes[0]
        self.assertIsInstance(figure, Figure)
        # North and South: green, yellow, red (3 segments each). East and West: red, green (2 each).
        self.assertEqual(len(ax.collections), 3 + 3 + 2 + 2)
        self.assertEqual(ax.get_xlim(), (0, 8))
        # y positions 0..3 from the bottom, so North (listed first) is drawn on top
        self.assertEqual([t.get_text() for t in ax.get_yticklabels()], ["West", "East", "South", "North"])
        self.assertEqual(legend_texts(ax), ["Green", "Yellow", "Red"])
        self.assertIn("Signal States", ax.get_title())
        self.assertIn("Fixed-Time", ax.get_title())
        self.assertIn("time", ax.get_xlabel().lower())

    def test_window_selects_seconds(self):
        # seconds 0,1,2 only: N and S are green all the time, E and W red all the time
        ax = plots.plot_signal_states(self.result, start=0, end=3).axes[0]
        self.assertEqual(len(ax.collections), 4)
        self.assertEqual(ax.get_xlim(), (0, 3))

    def test_window_in_the_middle(self):
        # seconds 2..5: N: green, yellow, red | S: same | E: red, green | W: same
        ax = plots.plot_signal_states(self.result, start=2, end=6).axes[0]
        self.assertEqual(len(ax.collections), 3 + 3 + 2 + 2)
        self.assertEqual(ax.get_xlim(), (2, 6))

    def test_end_beyond_history_is_clipped(self):
        ax = plots.plot_signal_states(self.result, start=5, end=1000).axes[0]
        self.assertEqual(ax.get_xlim(), (5, 8))
        self.assertEqual(len(ax.collections), 4)   # N,S red; E,W green

    def test_window_without_data_gives_a_message(self):
        figure = plots.plot_signal_states(self.result, start=50, end=60)
        self.assertEqual(len(figure.axes[0].collections), 0)
        self.assertIn("No simulation data", message_of(figure))

    def test_empty_history(self):
        figure = plots.plot_signal_states(empty_result())
        self.assertEqual(len(figure.axes[0].collections), 0)
        self.assertIn("No simulation data", message_of(figure))

    def test_missing_signals_field_gives_a_message(self):
        figure = plots.plot_signal_states(without_fields(self.result, "signals"))
        self.assertEqual(len(figure.axes[0].collections), 0)
        self.assertIn("not available", message_of(figure))

    def test_signal_that_is_not_a_signal_state_is_rejected(self):
        bad_history = tuple(replace(r, signals={d: "green" for d in Direction}) for r in self.result.history)
        with self.assertRaises(ValueError):
            plots.plot_signal_states(replace(self.result, history=bad_history))

    def test_invalid_window(self):
        for start, end in ((-1, None), (1.5, None), ("0", None), (True, None),
                           (0, 0), (5, 5), (5, 2), (0, 2.5), (0, "9"), (0, True)):
            with self.subTest(start=start, end=end):
                with self.assertRaises(ValueError):
                    plots.plot_signal_states(self.result, start, end)

    def test_invalid_result(self):
        for bad in (None, "x", 3):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    plots.plot_signal_states(bad)

    def test_a_gap_in_the_history_splits_a_segment(self):
        """If seconds are missing, a colour must not be drawn across the gap."""
        history = tuple(r for r in self.result.history if r.time != 1)    # drop second 1
        ax = plots.plot_signal_states(replace(self.result, history=history), start=0, end=3).axes[0]
        self.assertEqual(len(ax.collections), 4)                          # still 4 colour groups
        segments = sum(len(c.get_paths()) for c in ax.collections)
        self.assertEqual(segments, 8)                                     # 2 pieces per direction


# --------------------------------------------------------------------------- #
# 10. Arrivals and departures
# --------------------------------------------------------------------------- #
class TestArrivalsDepartures(unittest.TestCase):
    def test_two_cumulative_lines(self):
        figure = plots.plot_arrivals_departures(fixed_result())
        ax = figure.axes[0]
        self.assertIsInstance(figure, Figure)
        self.assertEqual(len(ax.lines), 2)
        self.assertEqual(list(ax.lines[0].get_ydata()), [1, 2, 3, 5, 5, 6, 6, 6, 6, 6])   # arrivals
        self.assertEqual(list(ax.lines[1].get_ydata()), [0, 0, 1, 2, 3, 4, 4, 4, 4, 4])   # departures
        self.assertEqual(legend_texts(ax), ["Cumulative arrivals", "Cumulative departures"])

    def test_departures_never_exceed_arrivals_in_the_plotted_data(self):
        ax = plots.plot_arrivals_departures(fixed_result()).axes[0]
        for arrived, departed in zip(ax.lines[0].get_ydata(), ax.lines[1].get_ydata()):
            self.assertLessEqual(departed, arrived)

    def test_titles_labels_and_axis(self):
        ax = plots.plot_arrivals_departures(adaptive_result()).axes[0]
        self.assertIn("Adaptive Multi-Agent", ax.get_title())
        self.assertIn("time", ax.get_xlabel().lower())
        self.assertIn("vehicles", ax.get_ylabel())
        self.assertEqual(ax.get_ylim()[0], 0)
        self.assertEqual(to_hex(ax.lines[1].get_color()), to_hex(plots.ADAPTIVE_COLOR))

    def test_no_traffic(self):
        ax = plots.plot_arrivals_departures(make_result([], steps=5)).axes[0]
        self.assertEqual(list(ax.lines[0].get_ydata()), [0] * 5)
        self.assertEqual(ax.get_ylim(), (0, 1.0))

    def test_empty_history(self):
        figure = plots.plot_arrivals_departures(empty_result())
        self.assertEqual(len(figure.axes[0].lines), 0)
        self.assertIn("No simulation data", message_of(figure))

    def test_missing_total_fields(self):
        for missing in ("total_arrived", "total_departed"):
            with self.subTest(missing=missing):
                figure = plots.plot_arrivals_departures(without_fields(fixed_result(), missing))
                self.assertEqual(len(figure.axes[0].lines), 0)
                self.assertIn("not available", message_of(figure))

    def test_missing_optional_signal_field_does_not_matter(self):
        ax = plots.plot_arrivals_departures(without_fields(fixed_result(), "signals")).axes[0]
        self.assertEqual(len(ax.lines), 2)

    def test_invalid_input(self):
        for bad in (None, "x", 3):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    plots.plot_arrivals_departures(bad)


# --------------------------------------------------------------------------- #
# Helpers: PNG export, figure lifecycle, architecture rule
# --------------------------------------------------------------------------- #
class TestFigureHelpers(unittest.TestCase):
    def test_png_bytes(self):
        figure = plots.plot_throughput(metrics_comparison())
        data = plots.figure_to_png_bytes(figure, dpi=50)
        self.assertIsInstance(data, bytes)
        self.assertTrue(data.startswith(b"\x89PNG"))

    def test_png_works_for_placeholder_figures(self):
        figure = plots.plot_arrivals_departures(empty_result())
        self.assertTrue(plots.figure_to_png_bytes(figure, dpi=50).startswith(b"\x89PNG"))

    def test_png_invalid_inputs(self):
        figure = plots.plot_throughput(metrics_comparison())
        for bad in (None, "figure", 3):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    plots.figure_to_png_bytes(bad)
        for bad_dpi in (0, -5, 1.5, True, "100"):
            with self.subTest(bad_dpi=bad_dpi):
                with self.assertRaises(ValueError):
                    plots.figure_to_png_bytes(figure, dpi=bad_dpi)

    def test_every_call_returns_a_new_independent_figure(self):
        comparison = metrics_comparison()
        first = plots.plot_throughput(comparison)
        second = plots.plot_throughput(comparison)
        self.assertIsNot(first, second)
        self.assertIsNot(first.axes[0], second.axes[0])

    def test_plotting_does_not_leave_pyplot_figures_open(self):
        before = plt.get_fignums()
        comparison = metrics_comparison()
        plots.plot_direction_average_queue(comparison)
        plots.plot_queue_over_time(fixed_result(), adaptive_result())
        plots.plot_signal_states(make_result([], steps=8, signals_for=signal_cycle))
        plots.plot_arrivals_departures(fixed_result())
        plots.plot_arrivals_departures(empty_result())
        self.assertEqual(plt.get_fignums(), before)

    def test_plots_do_not_modify_their_inputs(self):
        fixed, adaptive = fixed_result(), adaptive_result()
        comparison = metrics_comparison(fixed, adaptive)
        history_before, vehicles_before = fixed.history, fixed.vehicles
        rows_before = comparison.rows
        plots.plot_queue_over_time(fixed, adaptive)
        plots.plot_signal_states(fixed)
        plots.plot_arrivals_departures(fixed)
        plots.plot_average_waiting_time(comparison)
        plots.plot_direction_average_queue(comparison)
        self.assertEqual(fixed.history, history_before)
        self.assertEqual(fixed.vehicles, vehicles_before)
        self.assertEqual(comparison.rows, rows_before)

    def test_module_does_not_run_simulations_or_calculate_metrics(self):
        """Architecture rule: plots.py only draws. These names must not even be imported."""
        for forbidden in ("Simulator", "run_comparison", "TrafficGenerator", "TrafficSchedule",
                          "calculate_metrics", "compare_results", "evaluate_comparison", "Junction"):
            with self.subTest(name=forbidden):
                self.assertFalse(hasattr(plots, forbidden))

    def test_same_input_gives_the_same_plotted_data(self):
        a = plots.plot_queue_over_time(fixed_result(), adaptive_result()).axes[0]
        b = plots.plot_queue_over_time(fixed_result(), adaptive_result()).axes[0]
        for line_a, line_b in zip(a.lines, b.lines):
            self.assertEqual(list(line_a.get_ydata()), list(line_b.get_ydata()))


# --------------------------------------------------------------------------- #
# build_all_figures + real simulator data
# --------------------------------------------------------------------------- #
class TestWithRealSimulatorData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runs = run_comparison(cfg(120))
        cls.metrics = evaluate_comparison(cls.runs)
        cls.figures = plots.build_all_figures(cls.runs, cls.metrics)

    def test_build_all_figures_returns_every_chart(self):
        expected = {
            "average_waiting_time", "max_waiting_time", "average_queue_length", "max_queue_length",
            "throughput", "congestion_index", "direction_average_queue", "queue_over_time",
            "direction_queues_fixed_time", "direction_queues_adaptive",
            "signal_states_fixed_time", "signal_states_adaptive",
            "arrivals_departures_fixed_time", "arrivals_departures_adaptive",
        }
        self.assertEqual(set(self.figures), expected)
        for name, figure in self.figures.items():
            with self.subTest(figure=name):
                self.assertIsInstance(figure, Figure)
                self.assertEqual(len(figure.axes), 1)

    def test_bar_charts_show_the_metrics_numbers(self):
        heights = bar_heights(self.figures["throughput"].axes[0])
        self.assertAlmostEqual(heights[0], self.metrics.fixed_time.throughput_per_minute)
        self.assertAlmostEqual(heights[1], self.metrics.adaptive.throughput_per_minute)

    def test_time_series_use_the_simulation_history(self):
        ax = self.figures["queue_over_time"].axes[0]
        self.assertEqual(len(ax.lines), 2)
        self.assertEqual(len(ax.lines[0].get_xdata()), 120)
        expected = [sum(r.queue_lengths.values()) for r in self.runs.fixed_time.history]
        self.assertEqual(list(ax.lines[0].get_ydata()), expected)

    def test_signal_charts_are_drawn_and_default_window_is_clipped_to_the_run(self):
        for name in ("signal_states_fixed_time", "signal_states_adaptive"):
            with self.subTest(figure=name):
                ax = self.figures[name].axes[0]
                self.assertGreater(len(ax.collections), 0)
                self.assertEqual(ax.get_xlim(), (0, 120))     # default window (0, 300) clipped to 120 s

    def test_arrivals_charts_end_at_the_real_totals(self):
        ax = self.figures["arrivals_departures_adaptive"].axes[0]
        self.assertEqual(ax.lines[0].get_ydata()[-1], self.runs.adaptive.total_arrived)
        self.assertEqual(ax.lines[1].get_ydata()[-1], self.runs.adaptive.total_departed)

    def test_custom_signal_window(self):
        figures = plots.build_all_figures(self.runs, self.metrics, signal_window=(10, 40))
        self.assertEqual(figures["signal_states_fixed_time"].axes[0].get_xlim(), (10, 40))

    def test_build_all_figures_invalid_inputs(self):
        with self.assertRaises(ValueError):
            plots.build_all_figures(None, self.metrics)
        with self.assertRaises(ValueError):
            plots.build_all_figures(self.runs, None)
        with self.assertRaises(ValueError):
            plots.build_all_figures(self.runs.fixed_time, self.metrics)
        for bad in ((0,), [0, 10], "0-10", None):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    plots.build_all_figures(self.runs, self.metrics, signal_window=bad)
        with self.assertRaises(ValueError):
            plots.build_all_figures(self.runs, self.metrics, signal_window=(-1, 10))


if __name__ == "__main__":
    unittest.main()
