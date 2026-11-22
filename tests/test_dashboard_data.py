"""Tests for dashboard_data.py's read-only DataFrame and CSV helpers."""

import csv
import io
import unittest

import pandas as pd

from config import Direction, SimulationConfig
from controllers import ControllerDecision
from coordinator import Action
from dashboard_data import (
    comparison_metrics_frame,
    direction_csv_bytes,
    direction_metrics_frame,
    history_csv_bytes,
    history_frame,
    metrics_csv_bytes,
)
from junction import Phase, SignalState, Stage
from metrics import compare_metrics, calculate_metrics
from simulator import ControllerType, SimulationResult, StepRecord, VehicleRecord


FIXED = ControllerType.FIXED_TIME
ADAPTIVE = ControllerType.ADAPTIVE


def make_result(controller_type=FIXED, history_size=2):
    config = SimulationConfig(duration=history_size)
    vehicles = (
        VehicleRecord(1, Direction.NORTH, 0, 1),
        VehicleRecord(2, Direction.SOUTH, 1, None),
    )

    history = []
    for time in range(history_size):
        arrived = {direction: 0 for direction in Direction}
        departed = {direction: 0 for direction in Direction}
        if time == 0:
            arrived[Direction.NORTH] = 1
            departed[Direction.NORTH] = 0
        if time == 1:
            arrived[Direction.SOUTH] = 1
            departed[Direction.NORTH] = 1

        history.append(StepRecord(
            time=time,
            stage=Stage.GREEN,
            active_phase=Phase.NORTH_SOUTH if time == 0 else Phase.EAST_WEST,
            signals={
                Direction.NORTH: SignalState.GREEN if time == 0 else SignalState.RED,
                Direction.SOUTH: SignalState.GREEN if time == 0 else SignalState.RED,
                Direction.EAST: SignalState.RED if time == 0 else SignalState.GREEN,
                Direction.WEST: SignalState.RED if time == 0 else SignalState.GREEN,
            },
            queue_lengths={
                Direction.NORTH: 1 if time == 0 else 0,
                Direction.SOUTH: 0 if time == 0 else 1,
                Direction.EAST: 0,
                Direction.WEST: 0,
            },
            arrived=arrived,
            departed=departed,
            total_arrived=time + 1,
            total_departed=1 if time == 1 else 0,
            decision=ControllerDecision(Action.NONE, None, 0),
        ))

    name = "Fixed-Time" if controller_type is FIXED else "Adaptive Multi-Agent"
    return SimulationResult(config, controller_type, name, tuple(history), vehicles)


class TestDashboardData(unittest.TestCase):
    def setUp(self):
        self.fixed = make_result(FIXED)
        self.adaptive = make_result(ADAPTIVE)
        self.fixed_metrics = calculate_metrics(self.fixed)
        self.adaptive_metrics = calculate_metrics(self.adaptive)
        self.comparison = compare_metrics(self.fixed_metrics, self.adaptive_metrics)

    def test_comparison_metrics_frame(self):
        frame = comparison_metrics_frame(self.comparison)
        self.assertEqual(frame.shape, (7, 7))
        self.assertEqual(list(frame.columns), [
            "metric", "label", "unit", "fixed_time", "adaptive",
            "improvement_percent", "winner",
        ])
        self.assertEqual(frame.iloc[0]["metric"], "average_waiting_time")

    def test_direction_metrics_frame_values_and_rows(self):
        frame = direction_metrics_frame(self.comparison)
        self.assertEqual(frame.shape, (8, 6))
        north_fixed = frame[
            (frame["controller_name"] == "Fixed-Time")
            & (frame["direction"] == "North")
        ].iloc[0]
        self.assertEqual(north_fixed["vehicles_arrived"], 1)
        self.assertEqual(north_fixed["vehicles_departed"], 1)
        self.assertEqual(north_fixed["maximum_queue_length"], 1)

    def test_history_frame_columns_values_and_enum_conversion(self):
        frame = history_frame(self.fixed)
        self.assertEqual(frame.shape, (2, 22))
        self.assertEqual(frame.iloc[0]["time"], 0)
        self.assertEqual(frame.iloc[0]["stage"], "green")
        self.assertEqual(frame.iloc[0]["active_phase"], "North-South")
        self.assertEqual(frame.iloc[0]["north_signal"], "green")
        self.assertEqual(frame.iloc[0]["north_queue"], 1)
        self.assertEqual(frame.iloc[0]["north_arrived"], 1)

    def test_empty_history_has_expected_columns(self):
        empty = SimulationResult(
            SimulationConfig(duration=1), FIXED, "Fixed-Time", tuple(), tuple(),
        )
        frame = history_frame(empty)
        self.assertEqual(frame.shape, (0, 22))
        self.assertIn("west_departed", frame.columns)

    def test_missing_optional_direction_data_is_not_invented(self):
        record = StepRecord(
            time=0,
            stage=Stage.IDLE,
            active_phase=None,
            signals={},
            queue_lengths={},
            arrived={},
            departed={},
            total_arrived=0,
            total_departed=0,
            decision=ControllerDecision(Action.NONE, None, 0),
        )
        result = SimulationResult(
            SimulationConfig(duration=1), FIXED, "Fixed-Time", (record,), tuple(),
        )
        frame = history_frame(result)
        self.assertTrue(pd.isna(frame.iloc[0]["active_phase"]))
        self.assertTrue(frame.isna().any().any())

    def test_csv_exports_are_utf8_and_round_trip(self):
        history_bytes = history_csv_bytes(self.fixed)
        metrics_bytes = metrics_csv_bytes(self.comparison)
        direction_bytes = direction_csv_bytes(self.comparison)

        for csv_bytes in (history_bytes, metrics_bytes, direction_bytes):
            self.assertIsInstance(csv_bytes, bytes)
            csv_bytes.decode("utf-8")

        rows = list(csv.reader(io.StringIO(history_bytes.decode("utf-8"))))
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0][0], "time")

    def test_invalid_inputs_raise_clear_value_errors(self):
        invalid_values = (
            comparison_metrics_frame,
            direction_metrics_frame,
            metrics_csv_bytes,
            direction_csv_bytes,
        )
        for function in invalid_values:
            with self.subTest(function=function.__name__):
                with self.assertRaisesRegex(ValueError, "MetricsComparison"):
                    function(object())

        with self.assertRaisesRegex(ValueError, "SimulationResult"):
            history_frame(object())
        with self.assertRaisesRegex(ValueError, "SimulationResult"):
            history_csv_bytes(object())


if __name__ == "__main__":
    unittest.main()
