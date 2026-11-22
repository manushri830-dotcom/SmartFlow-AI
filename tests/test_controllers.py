"""Tests for controllers.py (common interface, FixedTimeController, AdaptiveController)."""

import itertools
import unittest
from dataclasses import FrozenInstanceError, replace
from typing import Callable, Dict, List, Optional, Tuple

from config import Direction, SimulationConfig
from controllers import (
    AdaptiveController,
    ControllerDecision,
    FixedTimeController,
    TrafficController,
)
from coordinator import Action, Reason
from junction import Junction, Phase, Stage
from traffic import Lane, Vehicle

_vehicle_ids = itertools.count(1)

# One log entry per second: (stage, active phase), recorded AFTER update() and
# BEFORE advance(), i.e. exactly what the signals show during that second.
LogEntry = Tuple[Stage, Optional[Phase]]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def make_lanes() -> Dict[Direction, Lane]:
    return {direction: Lane(direction) for direction in Direction}


def add_vehicles(lane: Lane, count: int, arrival_time: int = 0) -> None:
    for _ in range(count):
        lane.add_vehicle(Vehicle(vehicle_id=next(_vehicle_ids), arrival_time=arrival_time))


def run(
    controller: TrafficController,
    seconds: int,
    lanes: Optional[Dict[Direction, Lane]] = None,
    arrivals: Optional[Callable[[int], Dict[Direction, int]]] = None,
) -> Tuple[List[LogEntry], List[ControllerDecision]]:
    """A tiny stand-in for the future simulator. Works for ANY controller.

    Per second: arrivals -> controller.update -> discharge (1 vehicle / 2 s per
    green lane) -> junction.advance. Checks that every second is safe.
    """
    junction = controller.junction
    log: List[LogEntry] = []
    decisions: List[ControllerDecision] = []
    for t in range(seconds):
        if arrivals is not None and lanes is not None:
            for direction, count in arrivals(t).items():
                add_vehicles(lanes[direction], count, t)
        decisions.append(controller.update(t))
        log.append((junction.stage, junction.active_phase))
        if lanes is not None:
            for direction in junction.green_directions():
                if t % 2 == 0:
                    lanes[direction].remove_vehicle(t)
        assert junction.is_safe(), f"unsafe signals at t={t}: {junction.signal_states()}"
        junction.advance()
    return log, decisions


def green_runs(log: List[LogEntry]) -> List[Tuple[Phase, int]]:
    """Lengths of COMPLETED green runs: [(phase, seconds), ...].

    A run is closed by the first non-green second, so a run still going at the
    end of the log is left out (the test may have stopped in the middle of it).
    """
    runs: List[Tuple[Phase, int]] = []
    phase: Optional[Phase] = None
    length = 0
    for stage, current_phase in log:
        if stage is Stage.GREEN:
            if length and current_phase is phase:
                length += 1
            else:
                phase, length = current_phase, 1
        elif length:
            runs.append((phase, length))
            length = 0
    return runs


ALLOWED_NEXT = {
    Stage.IDLE: {Stage.IDLE, Stage.GREEN},
    Stage.GREEN: {Stage.GREEN, Stage.YELLOW},
    Stage.YELLOW: {Stage.YELLOW, Stage.ALL_RED},
    Stage.ALL_RED: {Stage.ALL_RED, Stage.GREEN, Stage.IDLE},
}


class SafeSequenceMixin:
    """Assertion shared by both controllers' tests."""

    def assert_safe_sequence(self, log: List[LogEntry]) -> None:
        for (prev_stage, prev_phase), (stage, phase) in zip(log, log[1:]):
            self.assertIn(stage, ALLOWED_NEXT[prev_stage], f"{prev_stage} -> {stage}")
            if prev_stage is Stage.GREEN and stage is Stage.GREEN:
                self.assertIs(prev_phase, phase, "green jumped straight to the other phase")


# --------------------------------------------------------------------------- #
# ControllerDecision and the base class
# --------------------------------------------------------------------------- #
class TestControllerDecision(unittest.TestCase):
    def test_holds_values(self):
        decision = ControllerDecision(Action.START, Phase.NORTH_SOUTH, 20)
        self.assertIs(decision.action, Action.START)
        self.assertIs(decision.phase, Phase.NORTH_SOUTH)
        self.assertEqual(decision.green_duration, 20)

    def test_is_frozen(self):
        decision = ControllerDecision(Action.NONE, None, 0)
        with self.assertRaises(FrozenInstanceError):
            decision.green_duration = 5


class TestBaseClass(unittest.TestCase):
    def test_cannot_instantiate_abstract_base(self):
        with self.assertRaises(TypeError):
            TrafficController(Junction(SimulationConfig()))

    def test_both_controllers_share_the_interface(self):
        junction = Junction(SimulationConfig())
        fixed = FixedTimeController(junction)
        adaptive = AdaptiveController(Junction(SimulationConfig()), make_lanes())
        for controller in (fixed, adaptive):
            self.assertIsInstance(controller, TrafficController)
            self.assertIsInstance(controller.junction, Junction)
            self.assertIsInstance(controller.name, str)
        self.assertNotEqual(fixed.name, adaptive.name)

    def test_junction_must_be_a_junction(self):
        for bad in (None, "junction", 42, SimulationConfig()):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    FixedTimeController(bad)
                with self.assertRaises(ValueError):
                    AdaptiveController(bad, make_lanes())

    def test_controller_exposes_the_same_junction_it_was_given(self):
        junction = Junction(SimulationConfig())
        self.assertIs(FixedTimeController(junction).junction, junction)


# --------------------------------------------------------------------------- #
# FixedTimeController
# --------------------------------------------------------------------------- #
class TestFixedTimeController(SafeSequenceMixin, unittest.TestCase):
    def setUp(self):
        self.config = SimulationConfig()
        self.junction = Junction(self.config)
        self.controller = FixedTimeController(self.junction)

    def test_uses_configured_fixed_green(self):
        self.assertEqual(self.controller.green_duration, 20)
        custom = replace(self.config, fixed_green=15)
        self.assertEqual(FixedTimeController(Junction(custom)).green_duration, 15)

    def test_first_update_starts_north_south(self):
        decision = self.controller.update(0)
        self.assertIs(decision.action, Action.START)
        self.assertIs(decision.phase, Phase.NORTH_SOUTH)
        self.assertEqual(decision.green_duration, 20)
        self.assertEqual(self.junction.green_directions(), (Direction.NORTH, Direction.SOUTH))
        self.assertEqual(self.junction.seconds_remaining, 20)

    def test_second_update_queues_east_west(self):
        self.controller.update(0)
        self.junction.advance()
        decision = self.controller.update(1)
        self.assertIs(decision.action, Action.QUEUE)
        self.assertIs(decision.phase, Phase.EAST_WEST)
        self.assertIs(self.junction.pending_phase, Phase.EAST_WEST)
        # the running green is NOT cut short
        self.assertEqual(self.junction.green_directions(), (Direction.NORTH, Direction.SOUTH))

    def test_does_not_requeue_while_a_phase_is_pending(self):
        _, decisions = run(self.controller, 24)
        actions = [d.action for d in decisions]
        self.assertEqual(actions[0], Action.START)
        self.assertEqual(actions[1], Action.QUEUE)
        self.assertTrue(all(a is Action.NONE for a in actions[2:24]))

    def test_exact_signal_timeline_with_no_traffic(self):
        log, _ = run(self.controller, 48)
        expected = (
            [(Stage.GREEN, Phase.NORTH_SOUTH)] * 20
            + [(Stage.YELLOW, Phase.NORTH_SOUTH)] * 3
            + [(Stage.ALL_RED, Phase.NORTH_SOUTH)] * 1
            + [(Stage.GREEN, Phase.EAST_WEST)] * 20
            + [(Stage.YELLOW, Phase.EAST_WEST)] * 3
            + [(Stage.ALL_RED, Phase.EAST_WEST)] * 1
        )
        self.assertEqual(log, expected)

    def test_cycle_repeats_every_24_seconds(self):
        log, _ = run(self.controller, 24 * 5)
        for i in range(24 * 3):
            self.assertEqual(log[i], log[i + 48], f"timeline differs at second {i}")

    def test_keeps_cycling_forever_without_traffic_and_stays_safe(self):
        log, _ = run(self.controller, 1000)   # run() asserts safety every second
        self.assert_safe_sequence(log)
        self.assertNotIn(Stage.IDLE, [stage for stage, _ in log])

    def test_every_green_lasts_exactly_fixed_green_and_phases_alternate(self):
        log, _ = run(self.controller, 600)
        runs = green_runs(log)
        self.assertGreaterEqual(len(runs), 20)
        for index, (phase, length) in enumerate(runs):
            self.assertEqual(length, 20)
            expected = Phase.NORTH_SOUTH if index % 2 == 0 else Phase.EAST_WEST
            self.assertIs(phase, expected)

    def test_custom_fixed_green_changes_the_timeline(self):
        config = replace(self.config, fixed_green=12, yellow_duration=2, all_red_duration=2)
        controller = FixedTimeController(Junction(config))
        log, _ = run(controller, 16 * 2)
        expected = (
            [(Stage.GREEN, Phase.NORTH_SOUTH)] * 12
            + [(Stage.YELLOW, Phase.NORTH_SOUTH)] * 2
            + [(Stage.ALL_RED, Phase.NORTH_SOUTH)] * 2
            + [(Stage.GREEN, Phase.EAST_WEST)] * 12
            + [(Stage.YELLOW, Phase.EAST_WEST)] * 2
            + [(Stage.ALL_RED, Phase.EAST_WEST)] * 2
        )
        self.assertEqual(log, expected)

    def test_ignores_traffic_completely(self):
        """Same signals whether the lanes are empty or jammed: the controller is blind."""
        empty_log, _ = run(FixedTimeController(Junction(self.config)), 200)

        lanes = make_lanes()
        busy_log, _ = run(
            FixedTimeController(Junction(self.config)),
            200,
            lanes=lanes,
            arrivals=lambda t: {Direction.EAST: 3, Direction.WEST: 3},
        )
        self.assertEqual(empty_log, busy_log)

    def test_all_directions_get_green_and_never_together(self):
        junction = self.junction
        seen = set()
        for t in range(100):
            self.controller.update(t)
            greens = junction.green_directions()
            seen.update(greens)
            ns = {Direction.NORTH, Direction.SOUTH} & set(greens)
            ew = {Direction.EAST, Direction.WEST} & set(greens)
            self.assertFalse(ns and ew, f"conflicting greens at t={t}")
            junction.advance()
        self.assertEqual(seen, set(Direction))

    def test_joins_a_junction_that_is_already_running(self):
        """If EAST_WEST is already green, the controller queues NORTH_SOUTH next."""
        self.junction.start_phase(Phase.EAST_WEST, 10)
        decision = self.controller.update(0)
        self.assertIs(decision.action, Action.QUEUE)
        self.assertIs(decision.phase, Phase.NORTH_SOUTH)
        self.assertIs(self.junction.active_phase, Phase.EAST_WEST)

    def test_rejects_invalid_time(self):
        for bad in (-1, 1.5, True, None, "5"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    self.controller.update(bad)
        self.assertTrue(self.junction.is_idle)   # a rejected call changes nothing


# --------------------------------------------------------------------------- #
# AdaptiveController: construction
# --------------------------------------------------------------------------- #
class TestAdaptiveConstruction(unittest.TestCase):
    def setUp(self):
        self.junction = Junction(SimulationConfig())

    def test_builds_four_agents_and_a_coordinator(self):
        controller = AdaptiveController(self.junction, make_lanes())
        self.assertIs(controller.coordinator.junction, self.junction)
        self.assertEqual(set(controller.coordinator.agents), set(Direction))

    def test_agents_use_the_given_lanes(self):
        lanes = make_lanes()
        controller = AdaptiveController(self.junction, lanes)
        for direction, agent in controller.coordinator.agents.items():
            self.assertIs(agent.lane, lanes[direction])

    def test_lanes_property_returns_a_copy(self):
        lanes = make_lanes()
        controller = AdaptiveController(self.junction, lanes)
        copy = controller.lanes
        self.assertEqual(copy, lanes)
        copy.clear()
        self.assertEqual(len(controller.lanes), 4)

    def test_default_and_custom_starvation_threshold(self):
        self.assertEqual(AdaptiveController(self.junction, make_lanes()).coordinator.starvation_threshold, 88)
        custom = AdaptiveController(self.junction, make_lanes(), starvation_threshold=50)
        self.assertEqual(custom.coordinator.starvation_threshold, 50)

    def test_weights_reach_the_agents(self):
        controller = AdaptiveController(self.junction, make_lanes(), queue_weight=2.0, wait_weight=0.25)
        for agent in controller.coordinator.agents.values():
            self.assertEqual(agent.queue_weight, 2.0)
            self.assertEqual(agent.wait_weight, 0.25)

    def test_rejects_invalid_arguments(self):
        lanes = make_lanes()
        missing = {d: lane for d, lane in lanes.items() if d is not Direction.WEST}
        extra = dict(lanes)
        extra["up"] = Lane(Direction.NORTH)
        not_a_lane = dict(lanes)
        not_a_lane[Direction.EAST] = "lane"
        swapped = dict(lanes)
        swapped[Direction.NORTH], swapped[Direction.SOUTH] = lanes[Direction.SOUTH], lanes[Direction.NORTH]

        for name, bad in (("None", None), ("list", list(lanes.values())), ("missing", missing),
                          ("extra", extra), ("not a Lane", not_a_lane), ("swapped", swapped)):
            with self.subTest(case=name):
                with self.assertRaises(ValueError):
                    AdaptiveController(self.junction, bad)

        with self.assertRaises(ValueError):
            AdaptiveController(self.junction, lanes, starvation_threshold=0)
        with self.assertRaises(ValueError):
            AdaptiveController(self.junction, lanes, queue_weight=0, wait_weight=0)
        with self.assertRaises(ValueError):
            AdaptiveController(self.junction, lanes, queue_weight=-1)

    def test_last_decision_is_none_before_first_update(self):
        self.assertIsNone(AdaptiveController(self.junction, make_lanes()).last_coordinator_decision)


# --------------------------------------------------------------------------- #
# AdaptiveController: behaviour
# --------------------------------------------------------------------------- #
class TestAdaptiveBehaviour(SafeSequenceMixin, unittest.TestCase):
    def setUp(self):
        self.config = SimulationConfig()
        self.junction = Junction(self.config)
        self.lanes = make_lanes()
        self.controller = AdaptiveController(self.junction, self.lanes)

    def test_no_demand_means_no_green(self):
        log, decisions = run(self.controller, 100, lanes=self.lanes)
        self.assertTrue(all(d.action is Action.NONE for d in decisions))
        self.assertTrue(all(d.phase is None for d in decisions))
        self.assertTrue(all(stage is Stage.IDLE for stage, _ in log))
        self.assertEqual(self.controller.last_coordinator_decision.reason, Reason.NO_DEMAND)

    def test_serves_the_only_phase_with_demand(self):
        add_vehicles(self.lanes[Direction.EAST], 2)
        decision = self.controller.update(0)
        self.assertIs(decision.action, Action.START)
        self.assertIs(decision.phase, Phase.EAST_WEST)
        self.assertEqual(decision.green_duration, self.config.min_green)
        self.assertEqual(self.junction.green_directions(), (Direction.EAST, Direction.WEST))

    def test_picks_the_more_urgent_phase(self):
        add_vehicles(self.lanes[Direction.NORTH], 6)
        add_vehicles(self.lanes[Direction.EAST], 2)
        decision = self.controller.update(0)
        self.assertIs(decision.phase, Phase.NORTH_SOUTH)
        self.assertIs(self.controller.last_coordinator_decision.reason, Reason.URGENCY)

    def test_green_duration_follows_queue_length(self):
        add_vehicles(self.lanes[Direction.SOUTH], 15)   # 15 vehicles * 2 s = 30 s
        decision = self.controller.update(0)
        self.assertEqual(decision.green_duration, 30)

    def test_decision_mirrors_the_coordinator_decision(self):
        add_vehicles(self.lanes[Direction.NORTH], 4)
        decision = self.controller.update(0)
        full = self.controller.last_coordinator_decision
        self.assertEqual(
            (decision.action, decision.phase, decision.green_duration),
            (full.action, full.phase, full.green_duration),
        )

    def test_running_green_is_never_interrupted(self):
        add_vehicles(self.lanes[Direction.NORTH], 20)
        self.controller.update(0)
        self.junction.advance()
        add_vehicles(self.lanes[Direction.EAST], 50, arrival_time=1)   # East suddenly very busy
        decision = self.controller.update(1)
        self.assertIs(decision.action, Action.NONE)
        self.assertIs(self.controller.last_coordinator_decision.reason, Reason.GREEN_IN_PROGRESS)
        self.assertEqual(self.junction.green_directions(), (Direction.NORTH, Direction.SOUTH))

    def test_serves_both_phases_when_both_have_demand(self):
        add_vehicles(self.lanes[Direction.NORTH], 3)
        add_vehicles(self.lanes[Direction.EAST], 3)
        log, _ = run(self.controller, 60, lanes=self.lanes)
        self.assertEqual(self.lanes[Direction.NORTH].total_departed, 3)
        self.assertEqual(self.lanes[Direction.EAST].total_departed, 3)
        phases = [phase for stage, phase in log if stage is Stage.GREEN]
        self.assertIs(phases[0], Phase.NORTH_SOUTH)   # tie on an idle junction -> NORTH_SOUTH first
        self.assertIn(Phase.EAST_WEST, phases)
        self.assert_safe_sequence(log)

    def test_responds_to_demand_that_appears_later(self):
        """Quiet junction, then East traffic appears: East must get a green."""
        arrivals = lambda t: {Direction.EAST: 1} if t in (30, 31, 32) else {}
        log, _ = run(self.controller, 80, lanes=self.lanes, arrivals=arrivals)
        self.assertTrue(all(stage is Stage.IDLE for stage, _ in log[:30]))
        self.assertEqual(self.lanes[Direction.EAST].total_departed, 3)

    def test_green_extends_while_demand_continues(self):
        """Heavy one-way traffic keeps the green longer than a single request."""
        arrivals = lambda t: {Direction.NORTH: 2}
        log, decisions = run(self.controller, 120, lanes=self.lanes, arrivals=arrivals)
        self.assertIn(Action.EXTEND, [d.action for d in decisions])

    def test_stays_safe_and_obeys_green_limits_under_mixed_load(self):
        # (every, offset): one vehicle arrives whenever (t + offset) % every == 0
        pattern = {
            Direction.NORTH: (3, 7),
            Direction.SOUTH: (5, 2),
            Direction.EAST: (2, 1),
            Direction.WEST: (4, 3),
        }

        def arrivals(t: int) -> Dict[Direction, int]:
            return {d: 1 for d, (every, offset) in pattern.items() if (t + offset) % every == 0}

        log, _ = run(self.controller, 1500, lanes=self.lanes, arrivals=arrivals)
        self.assert_safe_sequence(log)

        runs = green_runs(log)
        self.assertGreater(len(runs), 10)
        for _, length in runs:
            self.assertGreaterEqual(length, self.config.min_green)
            self.assertLessEqual(length, self.config.max_green)

    def test_starvation_guard_serves_a_lonely_vehicle(self):
        """One East vehicle waits while North/South are saturated: the guard must free it."""
        threshold = 60
        controller = AdaptiveController(self.junction, self.lanes, starvation_threshold=threshold)
        add_vehicles(self.lanes[Direction.EAST], 1, arrival_time=0)
        arrivals = lambda t: {Direction.NORTH: 1, Direction.SOUTH: 1}
        log, _ = run(controller, 400, lanes=self.lanes, arrivals=arrivals)

        east = self.lanes[Direction.EAST]
        self.assertEqual(east.total_departed, 1)
        waited = east.departed_vehicles[0].waiting_time(400)
        # worst case: green running when the threshold is reached (<= max_green),
        # then yellow + all-red, plus a couple of seconds of discharge alignment
        limit = threshold + self.config.max_green + self.config.yellow_duration + self.config.all_red_duration + 3
        self.assertLessEqual(waited, limit)
        self.assert_safe_sequence(log)

    def test_without_the_guard_the_same_vehicle_would_wait_much_longer(self):
        """Sanity check that the test above really tests the guard."""
        controller = AdaptiveController(self.junction, self.lanes, starvation_threshold=100_000)
        add_vehicles(self.lanes[Direction.EAST], 1, arrival_time=0)
        arrivals = lambda t: {Direction.NORTH: 1, Direction.SOUTH: 1}
        run(controller, 400, lanes=self.lanes, arrivals=arrivals)
        self.assertEqual(self.lanes[Direction.EAST].total_departed, 0)

    def test_rejects_invalid_time(self):
        for bad in (-1, 1.5, True, None, "5"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    self.controller.update(bad)


# --------------------------------------------------------------------------- #
# Both controllers through ONE loop (what the simulator will rely on)
# --------------------------------------------------------------------------- #
class TestSameLoopForBothControllers(SafeSequenceMixin, unittest.TestCase):
    @staticmethod
    def _build(kind: str, config: SimulationConfig):
        junction = Junction(config)
        lanes = make_lanes()
        if kind == "fixed":
            return FixedTimeController(junction), lanes
        return AdaptiveController(junction, lanes), lanes

    def test_identical_loop_runs_both_safely_on_identical_arrivals(self):
        config = SimulationConfig()
        schedule = {
            t: {d: (1 if (t * (i + 3)) % 7 == 0 else 0) for i, d in enumerate(Direction)}
            for t in range(600)
        }
        arrivals = lambda t: schedule[t]

        departed = {}
        for kind in ("fixed", "adaptive"):
            with self.subTest(controller=kind):
                controller, lanes = self._build(kind, config)
                log, decisions = run(controller, 600, lanes=lanes, arrivals=arrivals)
                self.assert_safe_sequence(log)
                self.assertTrue(all(isinstance(d, ControllerDecision) for d in decisions))
                self.assertEqual(
                    sum(lane.total_arrived for lane in lanes.values()),
                    sum(sum(second.values()) for second in schedule.values()),
                )
                departed[kind] = sum(lane.total_departed for lane in lanes.values())
        self.assertGreater(departed["fixed"], 0)
        self.assertGreater(departed["adaptive"], 0)

    def test_adaptive_idles_where_fixed_keeps_cycling(self):
        """With no traffic: fixed keeps cycling, adaptive wastes no green."""
        config = SimulationConfig()
        fixed, _ = self._build("fixed", config)
        adaptive, lanes = self._build("adaptive", config)
        fixed_log, _ = run(fixed, 60)
        adaptive_log, _ = run(adaptive, 60, lanes=lanes)
        self.assertIn(Stage.GREEN, [stage for stage, _ in fixed_log])
        self.assertNotIn(Stage.GREEN, [stage for stage, _ in adaptive_log])


if __name__ == "__main__":
    unittest.main()
