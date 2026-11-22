"""Tests for metrics.py.

Two kinds of tests:
  * HAND-BUILT results: we write the vehicles ourselves, so every expected number
    can be calculated on paper (see the comment above SCENARIO_A).
  * REAL simulator results: we check accounting rules and determinism only.
    We never assert that the adaptive controller is better.
"""

import math
import unittest
from dataclasses import FrozenInstanceError, replace
from typing import Dict, List, Optional, Tuple

from config import Direction, SimulationConfig
from controllers import ControllerDecision
from coordinator import Action
from junction import Phase, SignalState, Stage
from metrics import (
    DEFAULT_CONGESTION_THRESHOLD,
    MetricComparison,
    MetricsComparison,
    SimulationMetrics,
    Winner,
    calculate_metrics,
    compare_metrics,
    compare_results,
    evaluate_comparison,
    percentage_improvement,
)
from simulator import (
    ControllerType,
    SimulationResult,
    Simulator,
    StepRecord,
    TrafficSchedule,
    VehicleRecord,
    run_comparison,
)

N, S, E, W = Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST
FIXED, ADAPTIVE = ControllerType.FIXED_TIME, ControllerType.ADAPTIVE
NAMES = {FIXED: "Fixed-Time", ADAPTIVE: "Adaptive Multi-Agent"}

VehicleSpec = Tuple[int, Direction, int, Optional[int]]   # (id, direction, arrival, departure)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def cfg(duration: int = 100, **changes) -> SimulationConfig:
    """The default config with a shorter duration (and any other changes)."""
    return replace(SimulationConfig(), duration=duration, **changes)


def make_result(
    vehicles: List[VehicleSpec],
    steps: int,
    duration: Optional[int] = None,
    controller_type: ControllerType = FIXED,
) -> SimulationResult:
    """Build a consistent SimulationResult by hand.

    ``steps`` = seconds simulated. The per-second history (queues, arrivals,
    departures, running totals) is derived from the vehicle list, so it always
    agrees with it. Signals/decisions are dummies: metrics never reads them.
    """
    duration = max(steps, 1) if duration is None else duration
    config = cfg(duration)
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
        history.append(
            StepRecord(
                time=t,
                stage=Stage.GREEN,
                active_phase=Phase.NORTH_SOUTH,
                signals={d: SignalState.RED for d in Direction},
                queue_lengths=dict(queue),
                arrived=arrived,
                departed=departed,
                total_arrived=total_arrived,
                total_departed=total_departed,
                decision=ControllerDecision(Action.NONE, None, 0),
            )
        )
    return SimulationResult(config, controller_type, NAMES[controller_type], tuple(history), records)


def corrupt_step(result: SimulationResult, index: int, **changes) -> SimulationResult:
    """Copy of ``result`` with one StepRecord altered (to test validation)."""
    history = list(result.history)
    history[index] = replace(history[index], **changes)
    return replace(result, history=tuple(history))


# SCENARIO_A: 10 seconds, 6 vehicles. Hand calculation:
#   id  dir  arrive  depart   wait
#    1   N     0       2       2
#    2   N     1       4       3
#    3   S     2       3       1
#    4   E     3     (none)    10-3 = 7   (still queued, measured up to T=10)
#    5   E     3     (none)    7
#    6   W     5       5       0          (arrives and leaves in the same second)
#
#   arrived 6, departed 4 (ids 1,2,3,6), remaining 2
#   total wait = 2+3+1+7+7+0 = 20   -> average 20/6, max 7
#   average of departed only = (2+3+1+0)/4 = 1.5
#   queue at END of each second t = 0..9:
#     N: 1,2,1,1,0,0,0,0,0,0   sum 5   max 2
#     S: 0,0,1,0,0,0,0,0,0,0   sum 1   max 1
#     E: 0,0,0,2,2,2,2,2,2,2   sum 14  max 2
#     W: all 0                 sum 0   max 0
#   all queues summed over time = 20 (equals total wait: vehicle-seconds identity)
#   total queue per second: 1,2,2,3,2,2,2,2,2,2 -> max 3, average 20/10 = 2.0
#   average per lane = 20/(10*4) = 0.5
#   throughput = 4 * 60 / 10 = 24.0 vehicles/minute
#   lane-seconds with queue >= 2: N 1 (t=1) + E 7 (t=3..9) = 8 -> 8/40 = 0.2
SCENARIO_A: List[VehicleSpec] = [
    (1, N, 0, 2),
    (2, N, 1, 4),
    (3, S, 2, 3),
    (4, E, 3, None),
    (5, E, 3, None),
    (6, W, 5, 5),
]


def scenario_a(controller_type: ControllerType = FIXED) -> SimulationResult:
    return make_result(SCENARIO_A, steps=10, controller_type=controller_type)


# --------------------------------------------------------------------------- #
# 1. Hand-calculated metrics
# --------------------------------------------------------------------------- #
class TestHandCalculatedMetrics(unittest.TestCase):
    def setUp(self):
        self.m = calculate_metrics(scenario_a(), congestion_threshold=2)

    def test_run_information(self):
        self.assertIs(self.m.controller_type, FIXED)
        self.assertEqual(self.m.controller_name, "Fixed-Time")
        self.assertEqual(self.m.duration, 10)
        self.assertEqual(self.m.steps_completed, 10)
        self.assertTrue(self.m.is_complete)

    def test_vehicle_accounting(self):
        self.assertEqual(self.m.total_arrived, 6)
        self.assertEqual(self.m.total_departed, 4)
        self.assertEqual(self.m.vehicles_remaining, 2)
        self.assertEqual(self.m.arrived_by_direction, {N: 2, S: 1, E: 2, W: 1})
        self.assertEqual(self.m.departed_by_direction, {N: 2, S: 1, E: 0, W: 1})

    def test_waiting_times(self):
        self.assertEqual(self.m.total_waiting_time, 20)
        self.assertAlmostEqual(self.m.average_waiting_time, 20 / 6)
        self.assertAlmostEqual(self.m.average_waiting_time_departed, 1.5)
        self.assertEqual(self.m.max_waiting_time, 7)

    def test_queue_lengths(self):
        self.assertAlmostEqual(self.m.average_queue_length, 0.5)
        self.assertAlmostEqual(self.m.average_total_queue_length, 2.0)
        self.assertEqual(self.m.max_queue_length, 2)
        self.assertEqual(self.m.max_total_queue_length, 3)

    def test_direction_wise_queues(self):
        avg = self.m.average_queue_by_direction
        self.assertAlmostEqual(avg[N], 0.5)
        self.assertAlmostEqual(avg[S], 0.1)
        self.assertAlmostEqual(avg[E], 1.4)
        self.assertAlmostEqual(avg[W], 0.0)
        self.assertEqual(self.m.max_queue_by_direction, {N: 2, S: 1, E: 2, W: 0})

    def test_average_per_lane_is_mean_of_direction_averages(self):
        avg = self.m.average_queue_by_direction
        self.assertAlmostEqual(self.m.average_queue_length, sum(avg.values()) / 4)
        self.assertAlmostEqual(self.m.average_total_queue_length, sum(avg.values()))

    def test_max_queue_is_largest_direction_max(self):
        self.assertEqual(self.m.max_queue_length, max(self.m.max_queue_by_direction.values()))

    def test_throughput(self):
        self.assertAlmostEqual(self.m.throughput_per_minute, 24.0)

    def test_congestion_and_unserved(self):
        self.assertEqual(self.m.congestion_threshold, 2)
        self.assertAlmostEqual(self.m.congestion_index, 0.2)
        self.assertAlmostEqual(self.m.unserved_ratio, 2 / 6)

    def test_default_threshold_is_used_when_not_given(self):
        m = calculate_metrics(scenario_a())
        self.assertEqual(m.congestion_threshold, DEFAULT_CONGESTION_THRESHOLD)
        self.assertEqual(m.congestion_index, 0.0)   # no lane ever reaches 10 vehicles

    def test_threshold_changes_only_congestion(self):
        loose = calculate_metrics(scenario_a(), congestion_threshold=1)
        # lane-seconds with queue >= 1: N 4 + S 1 + E 7 = 12 -> 12/40
        self.assertAlmostEqual(loose.congestion_index, 0.3)
        self.assertEqual(loose.average_waiting_time, self.m.average_waiting_time)
        self.assertEqual(loose.max_queue_length, self.m.max_queue_length)

    def test_vehicle_seconds_identity(self):
        """Total waiting time must equal the sum of all queue lengths over time."""
        queue_sum = sum(record.total_waiting for record in scenario_a().history)
        self.assertEqual(self.m.total_waiting_time, queue_sum)

    def test_remaining_equals_final_queues(self):
        final = scenario_a().final_queue_lengths
        self.assertEqual(self.m.vehicles_remaining, sum(final.values()))

    def test_agrees_with_simulation_result_properties(self):
        result = scenario_a()
        self.assertEqual(self.m.total_arrived, result.total_arrived)
        self.assertEqual(self.m.total_departed, result.total_departed)
        self.assertEqual(self.m.vehicles_remaining, result.total_waiting)

    def test_still_queued_vehicles_are_counted(self):
        """Ignoring the two stuck vehicles would wrongly give 1.5 instead of 20/6."""
        self.assertGreater(self.m.average_waiting_time, self.m.average_waiting_time_departed)

    def test_metrics_are_frozen(self):
        with self.assertRaises(FrozenInstanceError):
            self.m.total_arrived = 99


# --------------------------------------------------------------------------- #
# 2. Edge cases
# --------------------------------------------------------------------------- #
class TestEdgeCases(unittest.TestCase):
    def assert_all_zero(self, m: SimulationMetrics):
        self.assertEqual(m.total_arrived, 0)
        self.assertEqual(m.total_departed, 0)
        self.assertEqual(m.vehicles_remaining, 0)
        self.assertEqual(m.total_waiting_time, 0)
        self.assertEqual(m.average_waiting_time, 0.0)
        self.assertEqual(m.average_waiting_time_departed, 0.0)
        self.assertEqual(m.max_waiting_time, 0)
        self.assertEqual(m.average_queue_length, 0.0)
        self.assertEqual(m.average_total_queue_length, 0.0)
        self.assertEqual(m.max_queue_length, 0)
        self.assertEqual(m.max_total_queue_length, 0)
        self.assertEqual(m.throughput_per_minute, 0.0)
        self.assertEqual(m.congestion_index, 0.0)
        self.assertEqual(m.unserved_ratio, 0.0)
        self.assertEqual(set(m.average_queue_by_direction), set(Direction))
        self.assertEqual(set(m.max_queue_by_direction), set(Direction))
        self.assertTrue(all(v == 0 for v in m.average_queue_by_direction.values()))
        self.assertTrue(all(v == 0 for v in m.max_queue_by_direction.values()))
        self.assertTrue(all(v == 0 for v in m.arrived_by_direction.values()))
        self.assertTrue(all(v == 0 for v in m.departed_by_direction.values()))

    def test_empty_simulation_nothing_simulated(self):
        """No second simulated yet: no division by zero, everything is zero."""
        m = calculate_metrics(make_result([], steps=0, duration=10))
        self.assert_all_zero(m)
        self.assertEqual(m.steps_completed, 0)
        self.assertFalse(m.is_complete)

    def test_complete_run_without_any_traffic(self):
        m = calculate_metrics(make_result([], steps=5))
        self.assert_all_zero(m)
        self.assertTrue(m.is_complete)

    def test_incomplete_simulation_measures_up_to_last_simulated_second(self):
        # 4 of 10 seconds simulated; the vehicle arrived at 0 and never left -> waited 4 s.
        m = calculate_metrics(make_result([(1, N, 0, None)], steps=4, duration=10))
        self.assertFalse(m.is_complete)
        self.assertEqual(m.steps_completed, 4)
        self.assertEqual(m.duration, 10)
        self.assertEqual(m.average_waiting_time, 4.0)
        self.assertEqual(m.max_waiting_time, 4)
        self.assertEqual(m.vehicles_remaining, 1)
        self.assertEqual(m.throughput_per_minute, 0.0)
        self.assertEqual(m.max_queue_length, 1)
        self.assertEqual(m.average_queue_by_direction[N], 1.0)

    def test_vehicle_that_arrives_and_leaves_in_same_second(self):
        m = calculate_metrics(make_result([(1, N, 0, 0)], steps=1))
        self.assertEqual(m.average_waiting_time, 0.0)
        self.assertEqual(m.max_waiting_time, 0)
        self.assertEqual(m.max_queue_length, 0)
        self.assertEqual(m.throughput_per_minute, 60.0)   # 1 vehicle in 1 second

    def test_vehicle_arriving_in_last_second_and_never_leaving(self):
        m = calculate_metrics(make_result([(1, W, 4, None)], steps=5))
        self.assertEqual(m.max_waiting_time, 1)   # measured up to T = 5
        self.assertEqual(m.vehicles_remaining, 1)

    def test_vehicle_departing_in_last_second(self):
        m = calculate_metrics(make_result([(1, S, 0, 4)], steps=5))
        self.assertEqual(m.total_departed, 1)
        self.assertEqual(m.max_waiting_time, 4)
        self.assertEqual(m.vehicles_remaining, 0)

    def test_all_vehicles_served(self):
        m = calculate_metrics(make_result([(1, N, 0, 1), (2, E, 0, 2)], steps=5))
        self.assertEqual(m.vehicles_remaining, 0)
        self.assertEqual(m.unserved_ratio, 0.0)
        self.assertEqual(m.average_waiting_time, m.average_waiting_time_departed)

    def test_no_vehicle_served(self):
        m = calculate_metrics(make_result([(1, N, 0, None), (2, S, 1, None)], steps=4))
        self.assertEqual(m.total_departed, 0)
        self.assertEqual(m.average_waiting_time_departed, 0.0)   # no departed vehicle: no division by zero
        self.assertEqual(m.unserved_ratio, 1.0)
        self.assertEqual(m.average_waiting_time, (4 + 3) / 2)

    def test_throughput_per_minute_scales_with_run_length(self):
        # 30 vehicles all arrive at 0 and leave one per second; run lasts 120 s -> 15 per minute.
        vehicles = [(i + 1, N, 0, i) for i in range(30)]
        m = calculate_metrics(make_result(vehicles, steps=120))
        self.assertEqual(m.total_departed, 30)
        self.assertAlmostEqual(m.throughput_per_minute, 15.0)

    def test_congestion_index_is_between_zero_and_one(self):
        m = calculate_metrics(make_result([(i, N, 0, None) for i in range(1, 6)], steps=4), congestion_threshold=1)
        self.assertAlmostEqual(m.congestion_index, 0.25)   # only North is congested: 1 of 4 lanes
        self.assertTrue(0.0 <= m.congestion_index <= 1.0)

    def test_numpy_style_integers_are_accepted_as_threshold(self):
        import numpy as np
        m = calculate_metrics(scenario_a(), congestion_threshold=np.int64(2))
        self.assertAlmostEqual(m.congestion_index, 0.2)


# --------------------------------------------------------------------------- #
# 3. Invalid input
# --------------------------------------------------------------------------- #
class TestInvalidInput(unittest.TestCase):
    def test_result_must_be_a_simulation_result(self):
        for bad in (None, "result", 42, {}, scenario_a().history):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    calculate_metrics(bad)

    def test_bad_congestion_threshold(self):
        for bad in (0, -1, 1.5, True, "5", None, float("nan")):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    calculate_metrics(scenario_a(), congestion_threshold=bad)

    def test_history_longer_than_duration(self):
        bad = replace(scenario_a(), config=cfg(3))
        with self.assertRaises(ValueError):
            calculate_metrics(bad)

    def test_duplicate_vehicle_ids(self):
        result = scenario_a()
        bad = replace(result, vehicles=result.vehicles + (result.vehicles[0],))
        with self.assertRaises(ValueError):
            calculate_metrics(bad)

    def test_vehicle_that_is_not_a_vehicle_record(self):
        bad = replace(scenario_a(), vehicles=scenario_a().vehicles + ("car",))
        with self.assertRaises(ValueError):
            calculate_metrics(bad)

    def test_bad_vehicle_id(self):
        base = make_result([], steps=3)
        for bad_id in (0, -1, 1.5, True):
            with self.subTest(bad_id=bad_id):
                bad = replace(base, vehicles=(VehicleRecord(bad_id, N, 0, None),))
                with self.assertRaises(ValueError):
                    calculate_metrics(bad)

    def test_bad_vehicle_direction(self):
        bad = replace(make_result([], steps=3), vehicles=(VehicleRecord(1, "North", 0, None),))
        with self.assertRaises(ValueError):
            calculate_metrics(bad)

    def test_vehicle_arrives_after_simulation_end(self):
        bad = replace(make_result([], steps=3), vehicles=(VehicleRecord(1, N, 3, None),))
        with self.assertRaises(ValueError):
            calculate_metrics(bad)

    def test_vehicle_in_a_result_with_no_history(self):
        bad = replace(make_result([], steps=0, duration=5), vehicles=(VehicleRecord(1, N, 0, None),))
        with self.assertRaises(ValueError):
            calculate_metrics(bad)

    def test_non_integer_arrival_time(self):
        bad = replace(make_result([], steps=3), vehicles=(VehicleRecord(1, N, 1.5, None),))
        with self.assertRaises(ValueError):
            calculate_metrics(bad)

    def test_departure_before_arrival(self):
        bad = replace(make_result([], steps=6), vehicles=(VehicleRecord(1, N, 3, 1),))
        with self.assertRaises(ValueError):
            calculate_metrics(bad)

    def test_departure_after_simulation_end(self):
        bad = replace(make_result([], steps=3), vehicles=(VehicleRecord(1, N, 0, 3),))
        with self.assertRaises(ValueError):
            calculate_metrics(bad)

    def test_history_out_of_order(self):
        with self.assertRaises(ValueError):
            calculate_metrics(corrupt_step(scenario_a(), 3, time=5))

    def test_history_entry_that_is_not_a_step_record(self):
        result = scenario_a()
        bad = replace(result, history=result.history[:-1] + ("step",))
        with self.assertRaises(ValueError):
            calculate_metrics(bad)

    def test_history_arrivals_disagree_with_vehicles(self):
        result = scenario_a()
        wrong = dict(result.history[0].arrived)
        wrong[N] = 5
        with self.assertRaises(ValueError):
            calculate_metrics(corrupt_step(result, 0, arrived=wrong))

    def test_history_departures_disagree_with_vehicles(self):
        result = scenario_a()
        wrong = dict(result.history[2].departed)
        wrong[E] = 1
        with self.assertRaises(ValueError):
            calculate_metrics(corrupt_step(result, 2, departed=wrong))

    def test_queue_length_does_not_balance(self):
        result = scenario_a()
        wrong = dict(result.history[4].queue_lengths)
        wrong[N] = 99
        with self.assertRaises(ValueError):
            calculate_metrics(corrupt_step(result, 4, queue_lengths=wrong))

    def test_missing_direction_in_history(self):
        result = scenario_a()
        wrong = {d: v for d, v in result.history[1].queue_lengths.items() if d is not W}
        with self.assertRaises(ValueError):
            calculate_metrics(corrupt_step(result, 1, queue_lengths=wrong))

    def test_negative_or_non_integer_counts_in_history(self):
        result = scenario_a()
        for bad_value in (-1, 1.5, True, None):
            with self.subTest(bad_value=bad_value):
                wrong = dict(result.history[1].queue_lengths)
                wrong[S] = bad_value
                with self.assertRaises(ValueError):
                    calculate_metrics(corrupt_step(result, 1, queue_lengths=wrong))

    def test_cumulative_totals_do_not_match(self):
        with self.assertRaises(ValueError):
            calculate_metrics(corrupt_step(scenario_a(), 9, total_arrived=7))
        with self.assertRaises(ValueError):
            calculate_metrics(corrupt_step(scenario_a(), 9, total_departed=0))

    def test_calculate_metrics_does_not_modify_the_result(self):
        result = scenario_a()
        history, vehicles = result.history, result.vehicles
        calculate_metrics(result)
        self.assertEqual(result.history, history)
        self.assertEqual(result.vehicles, vehicles)


# --------------------------------------------------------------------------- #
# 4. Percentage improvement
# --------------------------------------------------------------------------- #
class TestPercentageImprovement(unittest.TestCase):
    def test_lower_is_better(self):
        self.assertAlmostEqual(percentage_improvement(10, 5), 50.0)       # halved: 50 % better
        self.assertAlmostEqual(percentage_improvement(10, 15), -50.0)     # worse
        self.assertAlmostEqual(percentage_improvement(10, 0), 100.0)

    def test_higher_is_better(self):
        self.assertAlmostEqual(percentage_improvement(10, 15, higher_is_better=True), 50.0)
        self.assertAlmostEqual(percentage_improvement(10, 5, higher_is_better=True), -50.0)

    def test_equal_values_give_zero(self):
        self.assertEqual(percentage_improvement(7.5, 7.5), 0.0)
        self.assertEqual(percentage_improvement(7.5, 7.5, higher_is_better=True), 0.0)

    def test_zero_baseline_is_never_divided(self):
        self.assertEqual(percentage_improvement(0, 0), 0.0)
        self.assertIsNone(percentage_improvement(0, 3))
        self.assertIsNone(percentage_improvement(0, 3, higher_is_better=True))

    def test_invalid_values(self):
        for bad in (float("nan"), float("inf"), "5", None, True):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    percentage_improvement(bad, 1)
                with self.assertRaises(ValueError):
                    percentage_improvement(1, bad)

    def test_higher_is_better_must_be_a_bool(self):
        with self.assertRaises(ValueError):
            percentage_improvement(1, 2, higher_is_better="yes")


# --------------------------------------------------------------------------- #
# 5. Comparing two sets of metrics (crafted values, so the answers are known)
# --------------------------------------------------------------------------- #
def crafted_pair() -> Tuple[SimulationMetrics, SimulationMetrics]:
    """Fixed = SCENARIO_A metrics; adaptive = the same with hand-chosen different numbers."""
    fixed = calculate_metrics(scenario_a(FIXED), congestion_threshold=2)
    adaptive = replace(
        fixed,
        controller_type=ADAPTIVE,
        controller_name=NAMES[ADAPTIVE],
        average_waiting_time=fixed.average_waiting_time / 2,   # better: 50 %
        max_waiting_time=3,                                    # 7 -> 3: better
        # average_queue_length unchanged: tie
        max_queue_length=1,                                    # 2 -> 1: better: 50 %
        vehicles_remaining=0,                                  # 2 -> 0: better: 100 %
        throughput_per_minute=30.0,                            # 24 -> 30: better: 25 %
        congestion_index=0.3,                                  # 0.2 -> 0.3: WORSE: -50 %
    )
    return fixed, adaptive


class TestCompareMetrics(unittest.TestCase):
    def setUp(self):
        self.fixed, self.adaptive = crafted_pair()
        self.comparison = compare_metrics(self.fixed, self.adaptive)

    def test_returns_the_two_metrics_and_seven_rows(self):
        self.assertIsInstance(self.comparison, MetricsComparison)
        self.assertIs(self.comparison.fixed_time, self.fixed)
        self.assertIs(self.comparison.adaptive, self.adaptive)
        self.assertEqual(len(self.comparison.rows), 7)

    def test_improvement_percentages(self):
        row = self.comparison.row
        self.assertAlmostEqual(row("average_waiting_time").improvement_percent, 50.0)
        self.assertAlmostEqual(row("max_waiting_time").improvement_percent, 4 / 7 * 100)
        self.assertAlmostEqual(row("average_queue_length").improvement_percent, 0.0)
        self.assertAlmostEqual(row("max_queue_length").improvement_percent, 50.0)
        self.assertAlmostEqual(row("vehicles_remaining").improvement_percent, 100.0)
        self.assertAlmostEqual(row("throughput_per_minute").improvement_percent, 25.0)
        self.assertAlmostEqual(row("congestion_index").improvement_percent, -50.0)

    def test_winners_follow_the_numbers(self):
        row = self.comparison.row
        self.assertIs(row("average_waiting_time").winner, Winner.ADAPTIVE)
        self.assertIs(row("average_queue_length").winner, Winner.TIE)
        self.assertIs(row("throughput_per_minute").winner, Winner.ADAPTIVE)
        self.assertIs(row("congestion_index").winner, Winner.FIXED_TIME)   # adaptive is worse here

    def test_win_counts(self):
        self.assertEqual(self.comparison.count(Winner.ADAPTIVE), 5)
        self.assertEqual(self.comparison.count(Winner.FIXED_TIME), 1)
        self.assertEqual(self.comparison.count(Winner.TIE), 1)

    def test_identical_metrics_are_all_ties(self):
        same = replace(self.fixed, controller_type=ADAPTIVE, controller_name=NAMES[ADAPTIVE])
        comparison = compare_metrics(self.fixed, same)
        self.assertTrue(all(r.winner is Winner.TIE for r in comparison.rows))
        self.assertTrue(all(r.improvement_percent == 0.0 for r in comparison.rows))

    def test_zero_baseline_with_worse_adaptive_has_no_percentage_but_a_winner(self):
        fixed = replace(self.fixed, vehicles_remaining=0)
        adaptive = replace(self.adaptive, vehicles_remaining=2)
        row = compare_metrics(fixed, adaptive).row("vehicles_remaining")
        self.assertIsNone(row.improvement_percent)
        self.assertIs(row.winner, Winner.FIXED_TIME)

    def test_zero_baseline_for_higher_is_better_metric(self):
        fixed = replace(self.fixed, throughput_per_minute=0.0)
        row = compare_metrics(fixed, self.adaptive).row("throughput_per_minute")
        self.assertIsNone(row.improvement_percent)
        self.assertIs(row.winner, Winner.ADAPTIVE)

    def test_a_worse_adaptive_controller_is_reported_as_worse(self):
        """The module must never flatter the adaptive controller."""
        worse = replace(self.adaptive, average_waiting_time=self.fixed.average_waiting_time * 2)
        row = compare_metrics(self.fixed, worse).row("average_waiting_time")
        self.assertAlmostEqual(row.improvement_percent, -100.0)
        self.assertIs(row.winner, Winner.FIXED_TIME)

    def test_as_records(self):
        records = self.comparison.as_records()
        self.assertEqual(len(records), 7)
        first = records[0]
        self.assertEqual(
            set(first),
            {"metric", "label", "unit", "fixed_time", "adaptive", "improvement_percent", "winner"},
        )
        self.assertEqual(first["metric"], "average_waiting_time")
        self.assertEqual(first["winner"], "adaptive")

    def test_row_lookup_and_count_validation(self):
        with self.assertRaises(ValueError):
            self.comparison.row("nonsense")
        with self.assertRaises(ValueError):
            self.comparison.count("adaptive")

    def test_row_values_are_floats(self):
        for row in self.comparison.rows:
            self.assertIsInstance(row, MetricComparison)
            self.assertIsInstance(row.fixed_time, float)
            self.assertIsInstance(row.adaptive, float)

    def test_rejects_wrong_types(self):
        with self.assertRaises(ValueError):
            compare_metrics(None, self.adaptive)
        with self.assertRaises(ValueError):
            compare_metrics(self.fixed, "adaptive")

    def test_rejects_swapped_or_same_controller_types(self):
        with self.assertRaises(ValueError):
            compare_metrics(self.adaptive, self.fixed)
        with self.assertRaises(ValueError):
            compare_metrics(self.fixed, self.fixed)

    def test_rejects_different_run_lengths(self):
        with self.assertRaises(ValueError):
            compare_metrics(self.fixed, replace(self.adaptive, steps_completed=5))

    def test_rejects_different_traffic(self):
        different = dict(self.adaptive.arrived_by_direction)
        different[N] += 1
        with self.assertRaises(ValueError):
            compare_metrics(self.fixed, replace(self.adaptive, arrived_by_direction=different))

    def test_rejects_different_congestion_thresholds(self):
        with self.assertRaises(ValueError):
            compare_metrics(self.fixed, replace(self.adaptive, congestion_threshold=5))


# --------------------------------------------------------------------------- #
# 6. compare_results on hand-built results (fairness checks)
# --------------------------------------------------------------------------- #
class TestCompareResults(unittest.TestCase):
    def adaptive_version(self) -> SimulationResult:
        """Same vehicles as SCENARIO_A, but served differently."""
        vehicles = [
            (1, N, 0, 1), (2, N, 1, 2), (3, S, 2, 3),
            (4, E, 3, 4), (5, E, 3, 5), (6, W, 5, 5),
        ]
        return make_result(vehicles, steps=10, controller_type=ADAPTIVE)

    def test_fair_comparison_works(self):
        comparison = compare_results(scenario_a(FIXED), self.adaptive_version(), congestion_threshold=2)
        # adaptive served everyone: remaining 2 -> 0
        self.assertAlmostEqual(comparison.row("vehicles_remaining").improvement_percent, 100.0)
        self.assertIs(comparison.row("average_waiting_time").winner, Winner.ADAPTIVE)
        self.assertEqual(comparison.fixed_time.total_arrived, comparison.adaptive.total_arrived)

    def test_different_vehicles_are_rejected(self):
        other = make_result([(1, N, 0, 1)], steps=10, controller_type=ADAPTIVE)
        with self.assertRaises(ValueError):
            compare_results(scenario_a(FIXED), other)

    def test_same_ids_but_different_arrival_times_are_rejected(self):
        shifted = [(i, d, a + 1 if a < 9 else a, None) for i, d, a, _ in SCENARIO_A]
        with self.assertRaises(ValueError):
            compare_results(scenario_a(FIXED), make_result(shifted, steps=10, controller_type=ADAPTIVE))

    def test_different_configs_are_rejected(self):
        adaptive = replace(self.adaptive_version(), config=cfg(10, fixed_green=25))
        with self.assertRaises(ValueError):
            compare_results(scenario_a(FIXED), adaptive)

    def test_different_run_lengths_are_rejected(self):
        adaptive = make_result([], steps=3, duration=10, controller_type=ADAPTIVE)
        with self.assertRaises(ValueError):
            compare_results(make_result([], steps=5, duration=10, controller_type=FIXED), adaptive)

    def test_swapped_controllers_are_rejected(self):
        with self.assertRaises(ValueError):
            compare_results(self.adaptive_version(), scenario_a(FIXED))

    def test_invalid_results_are_rejected(self):
        with self.assertRaises(ValueError):
            compare_results(None, self.adaptive_version())
        with self.assertRaises(ValueError):
            compare_results(scenario_a(FIXED), "adaptive")

    def test_bad_threshold_is_rejected(self):
        with self.assertRaises(ValueError):
            compare_results(scenario_a(FIXED), self.adaptive_version(), congestion_threshold=0)

    def test_evaluate_comparison_rejects_wrong_type(self):
        with self.assertRaises(ValueError):
            evaluate_comparison("not a comparison")


# --------------------------------------------------------------------------- #
# 7. Real simulator results: accounting rules and determinism
#    (no test says the adaptive controller must win)
# --------------------------------------------------------------------------- #
class TestWithRealSimulator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = cfg(300)
        cls.runs = run_comparison(cls.config)
        cls.fixed = calculate_metrics(cls.runs.fixed_time)
        cls.adaptive = calculate_metrics(cls.runs.adaptive)

    def test_metrics_agree_with_the_schedule(self):
        total = self.runs.schedule.total_arrivals()
        self.assertGreater(total, 0)
        for m in (self.fixed, self.adaptive):
            self.assertEqual(m.total_arrived, total)
            for direction in Direction:
                self.assertEqual(m.arrived_by_direction[direction], self.runs.schedule.total_arrivals(direction))

    def test_both_controllers_receive_identical_traffic(self):
        self.assertEqual(self.fixed.arrived_by_direction, self.adaptive.arrived_by_direction)

    def test_vehicle_accounting_is_consistent(self):
        for result, m in ((self.runs.fixed_time, self.fixed), (self.runs.adaptive, self.adaptive)):
            with self.subTest(controller=m.controller_name):
                self.assertEqual(m.total_arrived, m.total_departed + m.vehicles_remaining)
                self.assertEqual(m.vehicles_remaining, sum(result.final_queue_lengths.values()))
                self.assertEqual(m.total_departed, sum(m.departed_by_direction.values()))
                self.assertEqual(m.total_arrived, sum(m.arrived_by_direction.values()))
                for direction in Direction:
                    self.assertLessEqual(m.departed_by_direction[direction], m.arrived_by_direction[direction])
                self.assertEqual(m.total_departed, result.total_departed)

    def test_vehicle_seconds_identity(self):
        for result, m in ((self.runs.fixed_time, self.fixed), (self.runs.adaptive, self.adaptive)):
            with self.subTest(controller=m.controller_name):
                self.assertEqual(m.total_waiting_time, sum(r.total_waiting for r in result.history))

    def test_throughput_formula(self):
        for m in (self.fixed, self.adaptive):
            self.assertAlmostEqual(m.throughput_per_minute, m.total_departed * 60 / self.config.duration)

    def test_values_are_in_valid_ranges(self):
        for m in (self.fixed, self.adaptive):
            self.assertTrue(m.is_complete)
            self.assertGreaterEqual(m.average_waiting_time, 0.0)
            self.assertGreaterEqual(m.max_waiting_time, m.average_waiting_time)
            self.assertGreaterEqual(m.max_queue_length, m.average_queue_length)
            self.assertTrue(0.0 <= m.congestion_index <= 1.0)
            self.assertTrue(0.0 <= m.unserved_ratio <= 1.0)

    def test_controller_information(self):
        self.assertIs(self.fixed.controller_type, FIXED)
        self.assertIs(self.adaptive.controller_type, ADAPTIVE)
        self.assertEqual(self.fixed.controller_name, self.runs.fixed_time.controller_name)

    def test_metrics_are_deterministic(self):
        self.assertEqual(calculate_metrics(self.runs.fixed_time), self.fixed)
        again = run_comparison(self.config)
        self.assertEqual(calculate_metrics(again.fixed_time), self.fixed)
        self.assertEqual(calculate_metrics(again.adaptive), self.adaptive)

    def test_evaluate_comparison_matches_manual_steps(self):
        evaluated = evaluate_comparison(self.runs)
        manual = compare_metrics(self.fixed, self.adaptive)
        self.assertEqual(evaluated, manual)
        self.assertEqual(len(evaluated.rows), 7)

    def test_winner_is_consistent_with_raw_numbers(self):
        """We do not predict who wins; we check the reported winner matches the numbers."""
        for row in evaluate_comparison(self.runs).rows:
            with self.subTest(metric=row.name):
                if math.isclose(row.fixed_time, row.adaptive, rel_tol=1e-9, abs_tol=1e-9):
                    self.assertIs(row.winner, Winner.TIE)
                elif row.adaptive < row.fixed_time:
                    self.assertIs(row.winner, Winner.FIXED_TIME if row.higher_is_better else Winner.ADAPTIVE)
                else:
                    self.assertIs(row.winner, Winner.ADAPTIVE if row.higher_is_better else Winner.FIXED_TIME)

    def test_no_traffic_run(self):
        quiet = replace(cfg(60), arrival_rate_north=0.0, arrival_rate_south=0.0,
                        arrival_rate_east=0.0, arrival_rate_west=0.0)
        comparison = evaluate_comparison(run_comparison(quiet))
        for m in (comparison.fixed_time, comparison.adaptive):
            self.assertEqual(m.total_arrived, 0)
            self.assertEqual(m.average_waiting_time, 0.0)
            self.assertEqual(m.throughput_per_minute, 0.0)
        self.assertTrue(all(r.winner is Winner.TIE for r in comparison.rows))
        self.assertTrue(all(r.improvement_percent == 0.0 for r in comparison.rows))

    def test_partial_run(self):
        schedule = TrafficSchedule.from_config(self.config)
        simulator = Simulator(self.config, FIXED, schedule)
        for _ in range(50):
            simulator.step()
        m = calculate_metrics(simulator.result())
        self.assertFalse(m.is_complete)
        self.assertEqual(m.steps_completed, 50)
        self.assertEqual(m.total_arrived, sum(
            schedule.arrival_count(t, d) for t in range(50) for d in Direction
        ))
        self.assertEqual(m.total_arrived, m.total_departed + m.vehicles_remaining)

    def test_simulator_that_has_not_started(self):
        schedule = TrafficSchedule.from_config(self.config)
        m = calculate_metrics(Simulator(self.config, FIXED, schedule).result())
        self.assertEqual(m.steps_completed, 0)
        self.assertEqual(m.total_arrived, 0)
        self.assertEqual(m.throughput_per_minute, 0.0)
        self.assertFalse(m.is_complete)

    def test_heavy_traffic_produces_congestion_accounting(self):
        heavy = replace(cfg(200), arrival_rate_north=0.6, arrival_rate_south=0.6,
                        arrival_rate_east=0.6, arrival_rate_west=0.6)
        run = run_comparison(heavy)
        for result in (run.fixed_time, run.adaptive):
            m = calculate_metrics(result)
            self.assertGreater(m.vehicles_remaining, 0)    # demand far above capacity
            self.assertEqual(m.total_arrived, m.total_departed + m.vehicles_remaining)
            self.assertEqual(m.total_waiting_time, sum(r.total_waiting for r in result.history))


if __name__ == "__main__":
    unittest.main()
