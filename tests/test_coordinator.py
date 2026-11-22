"""
tests/test_coordinator.py
=========================
Unit tests for coordinator.py (TrafficCoordinator).

Real project classes (Lane, SignalAgent, Junction, ...) are used throughout.
One mock-style spy is used where it makes a test clearer.

Uses only Python's built-in ``unittest``.

Run from the project folder (the one that contains config.py):
    python -m unittest tests.test_coordinator -v

Default config used by most tests:
    green 10-40 s, yellow 3 s, all-red 1 s, 0.5 vehicles/second (2 s per vehicle),
    agent weights 1.0 (queue) and 0.5 (wait), starvation threshold 88 s.
"""

import inspect
import re
import unittest
from unittest.mock import patch

import coordinator as coordinator_module
from agents import SignalAgent
from config import Direction, SimulationConfig
from coordinator import (
    Action,
    CoordinatorDecision,
    PhaseSummary,
    Reason,
    TrafficCoordinator,
    default_starvation_threshold,
)
from junction import Junction, Phase, SignalState, Stage
from traffic import Lane, TrafficGenerator, Vehicle

G, Y, R = SignalState.GREEN, SignalState.YELLOW, SignalState.RED
N, S, E, W = Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST

# Snapshot order is always (North, South, East, West).
NS_GREEN = (G, G, R, R)
NS_YELLOW = (Y, Y, R, R)
EW_GREEN = (R, R, G, G)
ALL_RED = (R, R, R, R)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def make_lane(direction, arrival_times=()):
    """A lane holding one vehicle per arrival time."""
    lane = Lane(direction)
    for vehicle_id, arrival_time in enumerate(arrival_times, start=1):
        lane.add_vehicle(Vehicle(vehicle_id=vehicle_id, arrival_time=arrival_time))
    return lane


class World:
    """A complete small system: four lanes, four agents, a junction and a coordinator."""

    def __init__(self, arrivals=None, config=None, starvation_threshold=None):
        arrivals = arrivals or {}
        self.config = config if config is not None else SimulationConfig()
        self.lanes = {d: make_lane(d, arrivals.get(d, ())) for d in Direction}
        self.agents = [SignalAgent(d, self.lanes[d], self.config) for d in Direction]
        self.junction = Junction(self.config)
        self.coordinator = TrafficCoordinator(
            self.junction, self.agents, starvation_threshold=starvation_threshold
        )

    def snapshot(self):
        """Signal colours as a tuple (North, South, East, West)."""
        return tuple(self.junction.signal_state(d) for d in Direction)

    def advance(self, seconds):
        """Advance the junction WITHOUT consulting the coordinator."""
        for _ in range(seconds):
            self.junction.advance()

    def summaries(self, current_time):
        """Per-phase summaries computed the same way the coordinator does."""
        requests = self.coordinator.collect_requests(current_time)
        return self.coordinator.summarise_phases(requests)

    def select(self, current_time):
        """What select_phase() answers right now."""
        return self.coordinator.select_phase(self.summaries(current_time))


def check_signal_history(test, history, max_green):
    """Safety checks shared by several tests, applied to a list of per-second snapshots."""
    allowed_changes = {(G, G), (Y, Y), (R, R), (G, Y), (Y, R), (R, G)}

    for second, colours in enumerate(history):
        ns_green = colours[0] is G or colours[1] is G
        ew_green = colours[2] is G or colours[3] is G
        test.assertFalse(ns_green and ew_green, msg=f"conflicting greens at second {second}: {colours}")

    for second in range(1, len(history)):
        before, now = history[second - 1], history[second]
        for index in range(4):
            test.assertIn(
                (before[index], now[index]), allowed_changes,
                msg=f"illegal change at second {second}: {before} -> {now}",
            )

    # No single green may last longer than max_green.
    run_phase, run_length = None, 0
    for second, colours in enumerate(history):
        if colours[0] is G:
            phase = Phase.NORTH_SOUTH
        elif colours[2] is G:
            phase = Phase.EAST_WEST
        else:
            phase = None
        if phase is not None and phase is run_phase:
            run_length += 1
        else:
            run_phase, run_length = phase, 1 if phase is not None else 0
        test.assertLessEqual(run_length, max_green, msg=f"green too long at second {second}")


def simulate(config, seconds, starvation_threshold=None):
    """A tiny stand-in for the future simulator, using the same per-second order.

    Each second: arrivals -> coordinator.update -> record -> discharge (1 vehicle
    per green lane every 2 seconds) -> junction.advance.
    """
    generator = TrafficGenerator(config)
    lanes = {d: Lane(d) for d in Direction}
    agents = [SignalAgent(d, lanes[d], config) for d in Direction]
    junction = Junction(config)
    coordinator = TrafficCoordinator(junction, agents, starvation_threshold=starvation_threshold)

    decisions, history, safe_flags = [], [], []
    for second in range(seconds):
        for direction, vehicles in generator.generate(second).items():
            for vehicle in vehicles:
                lanes[direction].add_vehicle(vehicle)
        decisions.append(coordinator.update(second))
        history.append(tuple(junction.signal_state(d) for d in Direction))
        safe_flags.append(junction.is_safe())
        if second % 2 == 0:
            for direction in junction.green_directions():
                lanes[direction].remove_vehicle(second)
        junction.advance()
    return coordinator, decisions, history, safe_flags, lanes


# --------------------------------------------------------------------------- #
# Initialisation
# --------------------------------------------------------------------------- #
class TestInitialization(unittest.TestCase):
    def test_valid_initialization_with_four_agents(self):
        world = World()
        coordinator = world.coordinator
        self.assertIs(coordinator.junction, world.junction)
        self.assertEqual(list(coordinator.agents), list(Direction))   # N, S, E, W order
        for agent in world.agents:
            self.assertIs(coordinator.agents[agent.direction], agent)

    def test_agents_may_be_given_in_any_order_and_any_iterable(self):
        world = World()
        TrafficCoordinator(world.junction, list(reversed(world.agents)))
        TrafficCoordinator(world.junction, tuple(world.agents))
        TrafficCoordinator(world.junction, (agent for agent in world.agents))

    def test_default_threshold_is_one_worst_case_cycle(self):
        coordinator = World().coordinator
        self.assertEqual(coordinator.starvation_threshold, 2 * (40 + 3 + 1))   # 88

    def test_default_threshold_is_derived_from_config(self):
        config = SimulationConfig(min_green=5, max_green=20, fixed_green=10)
        self.assertEqual(default_starvation_threshold(config), 2 * (20 + 3 + 1))
        self.assertEqual(World(config=config).coordinator.starvation_threshold, 48)

    def test_custom_threshold_is_used(self):
        self.assertEqual(World(starvation_threshold=30).coordinator.starvation_threshold, 30)

    def test_missing_agent_is_rejected(self):
        world = World()
        three_agents = [a for a in world.agents if a.direction is not W]
        with self.assertRaisesRegex(ValueError, "WEST"):
            TrafficCoordinator(world.junction, three_agents)
        with self.assertRaises(ValueError):
            TrafficCoordinator(world.junction, [])

    def test_duplicate_direction_is_rejected(self):
        world = World()
        extra_north = SignalAgent(N, make_lane(N), world.config)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            TrafficCoordinator(world.junction, world.agents + [extra_north])
        # Four agents, but North twice and West missing.
        replaced = [a for a in world.agents if a.direction is not W] + [extra_north]
        with self.assertRaises(ValueError):
            TrafficCoordinator(world.junction, replaced)

    def test_invalid_agent_is_rejected(self):
        world = World()
        valid_three = world.agents[:3]
        for bad in ("agent", None, 42, make_lane(W), object()):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    TrafficCoordinator(world.junction, valid_three + [bad])

    def test_agents_must_be_iterable(self):
        world = World()
        for bad in (None, 5):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    TrafficCoordinator(world.junction, bad)

    def test_invalid_junction_is_rejected(self):
        world = World()
        for bad in (None, "junction", SimulationConfig()):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    TrafficCoordinator(bad, world.agents)

    def test_invalid_threshold_is_rejected(self):
        world = World()
        for bad in (0, -5, 1.5, "10", True, float("nan")):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    TrafficCoordinator(world.junction, world.agents, starvation_threshold=bad)

    def test_agents_property_returns_a_copy(self):
        coordinator = World().coordinator
        coordinator.agents.clear()
        self.assertEqual(len(coordinator.agents), 4)


# --------------------------------------------------------------------------- #
# Collecting requests
# --------------------------------------------------------------------------- #
class TestRequestCollection(unittest.TestCase):
    def test_one_correct_request_per_direction(self):
        world = World({N: [0, 0, 0], S: [4], E: [], W: [2, 2]})
        requests = world.coordinator.collect_requests(10)
        self.assertEqual(list(requests), list(Direction))
        for direction, request in requests.items():
            self.assertIs(request.direction, direction)
            self.assertIs(request.phase, Phase.for_direction(direction))
        self.assertEqual(requests[N].queue_length, 3)
        self.assertEqual(requests[N].max_wait, 10)
        self.assertEqual(requests[S].queue_length, 1)
        self.assertEqual(requests[S].max_wait, 6)
        self.assertEqual(requests[E].queue_length, 0)
        self.assertEqual(requests[W].queue_length, 2)
        self.assertEqual(requests[W].max_wait, 8)

    def test_every_agent_is_asked_exactly_once_with_the_current_time(self):
        world = World({N: [0]})
        agent = world.agents[0]
        with patch.object(agent, "make_request", wraps=agent.make_request) as spy:
            world.coordinator.collect_requests(12)
        spy.assert_called_once_with(12)

    def test_invalid_time_is_rejected(self):
        coordinator = World().coordinator
        for bad in (-1, 1.5, "5", None, True):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    coordinator.collect_requests(bad)

    def test_collecting_changes_nothing(self):
        world = World({N: [0, 1], E: [0]})
        world.coordinator.collect_requests(5)
        self.assertTrue(world.junction.is_idle)
        self.assertEqual(world.lanes[N].queue_length, 2)
        self.assertEqual(world.lanes[E].queue_length, 1)


# --------------------------------------------------------------------------- #
# Urgency aggregation per phase
# --------------------------------------------------------------------------- #
class TestUrgencyAggregation(unittest.TestCase):
    ARRIVALS = {N: [0] * 8, S: [5, 5], E: [2], W: []}   # evaluated at t = 10

    def test_phase_urgency_is_the_sum_of_its_two_directions(self):
        summaries = World(self.ARRIVALS).summaries(10)
        # North: 8 + 0.5*10 = 13.0   South: 2 + 0.5*5 = 4.5   -> North/South = 17.5
        # East : 1 + 0.5*8  = 5.0    West : 0                 -> East/West   = 5.0
        self.assertAlmostEqual(summaries[Phase.NORTH_SOUTH].urgency, 17.5)
        self.assertAlmostEqual(summaries[Phase.EAST_WEST].urgency, 5.0)

    def test_phase_max_wait_is_the_longest_wait_of_its_directions(self):
        summaries = World(self.ARRIVALS).summaries(10)
        self.assertEqual(summaries[Phase.NORTH_SOUTH].max_wait, 10)
        self.assertEqual(summaries[Phase.EAST_WEST].max_wait, 8)

    def test_phase_requested_green_is_the_largest_request_of_the_phase(self):
        summaries = World(self.ARRIVALS).summaries(10)
        self.assertEqual(summaries[Phase.NORTH_SOUTH].requested_green, 16)   # North: 8 vehicles * 2 s
        self.assertEqual(summaries[Phase.EAST_WEST].requested_green, 10)     # min_green

    def test_phase_without_vehicles_has_no_demand(self):
        summary = World({N: [0, 0, 0]}).summaries(5)[Phase.EAST_WEST]
        self.assertIsInstance(summary, PhaseSummary)
        self.assertFalse(summary.has_demand)
        self.assertEqual(summary.urgency, 0.0)
        self.assertEqual(summary.max_wait, 0)
        self.assertEqual(summary.requested_green, 10)

    def test_phase_with_one_busy_direction_has_demand(self):
        summary = World({S: [0]}).summaries(5)[Phase.NORTH_SOUTH]
        self.assertTrue(summary.has_demand)

    def test_requested_green_is_clamped_to_the_junction_limits(self):
        summary = World({E: [0] * 100}).summaries(5)[Phase.EAST_WEST]
        self.assertEqual(summary.requested_green, 40)

    def test_summarise_requires_one_request_per_direction(self):
        world = World()
        requests = world.coordinator.collect_requests(0)
        del requests[W]
        with self.assertRaises(ValueError):
            world.coordinator.summarise_phases(requests)


# --------------------------------------------------------------------------- #
# Phase selection (no starvation involved)
# --------------------------------------------------------------------------- #
class TestPhaseSelection(unittest.TestCase):
    def test_highest_combined_urgency_wins(self):
        world = World(TestUrgencyAggregation.ARRIVALS)
        self.assertEqual(world.select(10), (Phase.NORTH_SOUTH, Reason.URGENCY))

    def test_east_west_can_win_too(self):
        world = World({N: [0] * 2, E: [0] * 10})   # t=10: North/South 7.0, East/West 15.0
        self.assertEqual(world.select(10), (Phase.EAST_WEST, Reason.URGENCY))

    def test_combined_urgency_counts_not_the_single_largest_queue(self):
        # East has the biggest single queue (8) but North + South together (5 + 5) beat it.
        world = World({N: [10] * 5, S: [10] * 5, E: [10] * 8})
        self.assertEqual(world.select(10), (Phase.NORTH_SOUTH, Reason.URGENCY))

    def test_tie_on_idle_junction_goes_to_north_south(self):
        world = World({N: [0] * 3, E: [0] * 3})
        summaries = world.summaries(5)
        self.assertEqual(summaries[Phase.NORTH_SOUTH].urgency, summaries[Phase.EAST_WEST].urgency)
        self.assertEqual(world.select(5), (Phase.NORTH_SOUTH, Reason.URGENCY))

    def test_tie_goes_to_the_phase_that_was_not_served_last(self):
        for last_served, expected in ((Phase.NORTH_SOUTH, Phase.EAST_WEST),
                                      (Phase.EAST_WEST, Phase.NORTH_SOUTH)):
            with self.subTest(last_served=last_served):
                world = World({N: [0] * 3, E: [0] * 3})
                world.junction.start_phase(last_served, 10)
                self.assertEqual(world.select(5), (expected, Reason.URGENCY))

    def test_selection_is_deterministic(self):
        world = World({N: [0] * 3, E: [0] * 3})
        answers = {world.select(5) for _ in range(20)}
        self.assertEqual(len(answers), 1)

    def test_no_demand_on_either_phase_selects_nothing(self):
        world = World()
        self.assertEqual(world.select(30), (None, Reason.NO_DEMAND))

    def test_empty_phase_is_never_chosen_while_the_other_has_demand(self):
        # On an idle junction North/South wins ties, yet here only East/West has vehicles.
        world = World({E: [0]})
        self.assertEqual(world.select(5), (Phase.EAST_WEST, Reason.URGENCY))
        # And the other way round, even when East/West was served last.
        world = World({N: [0]})
        world.junction.start_phase(Phase.NORTH_SOUTH, 10)
        self.assertEqual(world.select(5)[0], Phase.NORTH_SOUTH)


# --------------------------------------------------------------------------- #
# Starvation guard
# --------------------------------------------------------------------------- #
class TestStarvation(unittest.TestCase):
    """Threshold 30 s unless stated. At t=40: North has 30 fresh vehicles (urgency 30.5)."""

    BUSY_NORTH = [39] * 30

    def test_without_starvation_the_urgent_phase_wins(self):
        # East's single vehicle has waited 40 s (urgency 21.0) which is below North's 30.5.
        world = World({N: self.BUSY_NORTH, E: [0]}, starvation_threshold=100)
        self.assertEqual(world.select(40), (Phase.NORTH_SOUTH, Reason.URGENCY))

    def test_starving_phase_beats_a_more_urgent_phase(self):
        world = World({N: self.BUSY_NORTH, E: [0]}, starvation_threshold=30)
        self.assertEqual(world.select(40), (Phase.EAST_WEST, Reason.STARVATION))

    def test_threshold_is_inclusive(self):
        at_threshold = World({N: self.BUSY_NORTH, E: [10]}, starvation_threshold=30)   # waited 30
        below_threshold = World({N: self.BUSY_NORTH, E: [11]}, starvation_threshold=30)  # waited 29
        self.assertEqual(at_threshold.select(40), (Phase.EAST_WEST, Reason.STARVATION))
        self.assertEqual(below_threshold.select(40), (Phase.NORTH_SOUTH, Reason.URGENCY))

    def test_default_threshold_is_used_when_none_is_given(self):
        world = World({N: [99] * 30, E: [0]})   # default threshold 88; East has waited 100 s
        self.assertEqual(world.select(100), (Phase.EAST_WEST, Reason.STARVATION))

    def test_any_direction_of_a_phase_can_trigger_starvation(self):
        # Only West is old; East is fresh. The whole East/West phase is treated as starving.
        world = World({N: self.BUSY_NORTH, E: [39], W: [0]}, starvation_threshold=30)
        self.assertEqual(world.select(40), (Phase.EAST_WEST, Reason.STARVATION))

    def test_both_directions_of_one_phase_starving_is_not_a_conflict(self):
        world = World({N: [0], S: [0], E: [50]}, starvation_threshold=30)   # t=60: N,S waited 60
        phase, reason = world.select(60)
        self.assertEqual((phase, reason), (Phase.NORTH_SOUTH, Reason.STARVATION))

    def test_both_phases_starving_on_idle_junction_longest_wait_wins(self):
        north_older = World({N: [0], E: [10]}, starvation_threshold=30)   # t=60: 60 s vs 50 s
        east_older = World({N: [10], E: [0]}, starvation_threshold=30)
        self.assertEqual(north_older.select(60), (Phase.NORTH_SOUTH, Reason.STARVATION))
        self.assertEqual(east_older.select(60), (Phase.EAST_WEST, Reason.STARVATION))

    def test_equal_waits_are_settled_by_urgency(self):
        world = World({N: [0], E: [0] * 5}, starvation_threshold=30)   # both waited 60; 31.0 vs 35.0
        self.assertEqual(world.select(60), (Phase.EAST_WEST, Reason.STARVATION))

    def test_fully_equal_starvation_uses_the_tie_break_rule(self):
        world = World({N: [0], E: [0]}, starvation_threshold=30)
        self.assertEqual(world.select(60), (Phase.NORTH_SOUTH, Reason.STARVATION))   # idle -> NS
        world.junction.start_phase(Phase.NORTH_SOUTH, 10)
        self.assertEqual(world.select(60), (Phase.EAST_WEST, Reason.STARVATION))     # NS served last

    def test_when_both_phases_starve_they_take_turns(self):
        # North/South has the longer wait (60 s vs 50 s) but was served last, so East/West goes.
        world = World({N: [0], E: [10]}, starvation_threshold=30)
        world.junction.start_phase(Phase.NORTH_SOUTH, 10)
        self.assertEqual(world.select(60), (Phase.EAST_WEST, Reason.STARVATION))

    def test_starvation_never_selects_a_phase_without_vehicles(self):
        world = World({N: [0]}, starvation_threshold=30)
        phase, _ = world.select(100)
        self.assertEqual(phase, Phase.NORTH_SOUTH)

    def test_selected_phase_is_always_a_valid_phase(self):
        for arrivals in ({N: [0], E: [0]}, {S: [0], W: [0]}, {N: [0], S: [0], E: [0], W: [0]}):
            with self.subTest(arrivals=arrivals):
                world = World(arrivals, starvation_threshold=30)
                phase, _ = world.select(80)
                self.assertIn(phase, tuple(Phase))


# --------------------------------------------------------------------------- #
# update(): idle junction
# --------------------------------------------------------------------------- #
class TestUpdateIdle(unittest.TestCase):
    def test_no_demand_keeps_the_junction_idle(self):
        world = World()
        decision = world.coordinator.update(0)
        self.assertEqual(decision, CoordinatorDecision(Action.NONE, None, 0, Reason.NO_DEMAND))
        self.assertTrue(world.junction.is_idle)
        self.assertEqual(world.snapshot(), ALL_RED)

    def test_junction_stays_idle_for_many_seconds_without_demand(self):
        world = World()
        for second in range(30):
            world.coordinator.update(second)
            world.junction.advance()
        self.assertTrue(world.junction.is_idle)
        self.assertEqual(world.snapshot(), ALL_RED)

    def test_demand_on_one_phase_starts_that_phase(self):
        world = World({E: [0] * 3})
        decision = world.coordinator.update(5)
        self.assertEqual(decision, CoordinatorDecision(Action.START, Phase.EAST_WEST, 10, Reason.URGENCY))
        self.assertIs(world.junction.stage, Stage.GREEN)
        self.assertIs(world.junction.active_phase, Phase.EAST_WEST)
        self.assertEqual(world.snapshot(), EW_GREEN)
        self.assertTrue(world.junction.is_safe())

    def test_with_demand_on_both_phases_the_most_urgent_starts(self):
        world = World(TestUrgencyAggregation.ARRIVALS)
        decision = world.coordinator.update(10)
        self.assertEqual(decision.action, Action.START)
        self.assertIs(decision.phase, Phase.NORTH_SOUTH)
        self.assertEqual(decision.green_duration, 16)
        self.assertEqual(world.junction.seconds_remaining, 16)
        self.assertEqual(world.snapshot(), NS_GREEN)
        self.assertIsNone(world.junction.pending_phase)   # the other phase is NOT queued yet
        self.assertTrue(world.junction.is_safe())

    def test_started_green_is_clamped_to_max_green(self):
        world = World({E: [0] * 100})
        decision = world.coordinator.update(5)
        self.assertEqual(decision.green_duration, 40)
        self.assertEqual(world.junction.seconds_remaining, 40)

    def test_idle_junction_starts_when_demand_appears_later(self):
        world = World()
        for second in range(5):
            world.coordinator.update(second)
            world.junction.advance()
        world.lanes[W].add_vehicle(Vehicle(vehicle_id=1, arrival_time=5))
        decision = world.coordinator.update(5)
        self.assertEqual(decision.action, Action.START)
        self.assertIs(decision.phase, Phase.EAST_WEST)


# --------------------------------------------------------------------------- #
# update(): running green
# --------------------------------------------------------------------------- #
class TestUpdateGreen(unittest.TestCase):
    def make_world(self, arrivals, green_seconds=10, phase=Phase.NORTH_SOUTH, **kwargs):
        """A world whose junction is already serving ``phase`` (started outside the coordinator)."""
        world = World(arrivals, **kwargs)
        world.junction.start_phase(phase, green_seconds)
        return world

    def test_running_green_is_not_interrupted_by_a_more_urgent_phase(self):
        # East/West is far more urgent, but North/South is green and must finish its time.
        world = self.make_world({N: [0] * 5, E: [0] * 30})
        for second in range(9):                       # seconds 0..8 of a 10 s green
            decision = world.coordinator.update(second)
            self.assertEqual(decision.action, Action.NONE)
            self.assertEqual(decision.reason, Reason.GREEN_IN_PROGRESS)
            self.assertIs(decision.phase, Phase.NORTH_SOUTH)
            self.assertEqual(world.snapshot(), NS_GREEN)
            self.assertIsNone(world.junction.pending_phase)
            world.junction.advance()

    def test_green_lasts_exactly_its_granted_time(self):
        world = self.make_world({N: [0] * 5, E: [0] * 30})
        green_seconds = 0
        for second in range(20):
            world.coordinator.update(second)
            if world.snapshot() == NS_GREEN:
                green_seconds += 1
            world.junction.advance()
        self.assertEqual(green_seconds, 10)

    def test_last_second_of_green_does_not_queue_or_switch_directly(self):
        world = self.make_world({N: [0] * 5, E: [0] * 30})
        world.advance(9)                               # now 1 second of green remains
        decision = world.coordinator.update(9)
        self.assertEqual(decision.action, Action.NONE)
        self.assertIs(decision.phase, Phase.EAST_WEST)          # it WANTS East/West next...
        self.assertEqual(world.snapshot(), NS_GREEN)            # ...but nothing changed yet
        self.assertIsNone(world.junction.pending_phase)         # the queueing happens during clearance

    def test_green_is_extended_in_its_last_second_when_the_phase_still_wins(self):
        world = self.make_world({N: [0] * 30})
        world.advance(9)
        decision = world.coordinator.update(9)
        self.assertEqual(decision.action, Action.EXTEND)
        self.assertIs(decision.phase, Phase.NORTH_SOUTH)
        self.assertEqual(decision.green_duration, 30)           # 10 s so far, max 40 -> 30 more
        self.assertEqual(world.junction.seconds_remaining, 31)  # 1 left + 30 added
        self.assertTrue(world.junction.is_safe())

    def test_extended_green_lasts_exactly_max_green(self):
        world = self.make_world({N: [0] * 30})
        green_seconds = 0
        for second in range(60):
            world.coordinator.update(second)
            if world.snapshot() == NS_GREEN:
                green_seconds += 1
            elif green_seconds:
                break
            world.junction.advance()
        self.assertEqual(green_seconds, 40)

    def test_extension_is_limited_by_the_remaining_headroom(self):
        world = self.make_world({N: [0] * 30}, green_seconds=35)
        world.advance(34)
        decision = world.coordinator.update(34)
        self.assertEqual(decision.action, Action.EXTEND)
        self.assertEqual(decision.green_duration, 5)            # only 5 s below max_green
        self.assertEqual(world.junction.seconds_remaining, 6)

    def test_green_is_never_extended_beyond_max_green(self):
        world = self.make_world({N: [0] * 30}, green_seconds=40)
        world.advance(39)
        decision = world.coordinator.update(39)
        self.assertEqual(decision.action, Action.NONE)
        self.assertIs(decision.phase, Phase.NORTH_SOUTH)        # still wanted, but at the limit
        self.assertEqual(world.junction.seconds_remaining, 1)

    def test_no_extension_when_the_other_phase_wins(self):
        world = self.make_world({N: [0] * 5, E: [0] * 30})
        world.advance(9)
        decision = world.coordinator.update(9)
        self.assertEqual(decision.action, Action.NONE)
        self.assertEqual(world.junction.seconds_remaining, 1)

    def test_no_extension_on_a_tie(self):
        world = self.make_world({N: [0] * 3, E: [0] * 3})
        world.advance(9)
        decision = world.coordinator.update(9)
        self.assertEqual(decision.action, Action.NONE)
        self.assertIs(decision.phase, Phase.EAST_WEST)          # the phase not served last gets the turn

    def test_no_extension_when_the_other_phase_is_starving(self):
        # North/South is the more urgent phase, but East's vehicle has waited 9 s >= threshold 5.
        world = self.make_world({N: [8] * 30, E: [0]}, starvation_threshold=5)
        world.advance(9)
        decision = world.coordinator.update(9)
        self.assertEqual(decision.action, Action.NONE)
        self.assertIs(decision.phase, Phase.EAST_WEST)
        self.assertEqual(decision.reason, Reason.STARVATION)

    def test_no_extension_when_the_junction_already_has_a_queued_phase(self):
        world = self.make_world({N: [0] * 30})
        world.junction.start_phase(Phase.EAST_WEST, 10)         # queued by someone else
        world.advance(9)
        decision = world.coordinator.update(9)
        self.assertEqual(decision.action, Action.NONE)
        self.assertIs(world.junction.pending_phase, Phase.EAST_WEST)
        self.assertEqual(world.junction.seconds_remaining, 1)

    def test_last_second_with_no_demand_does_nothing(self):
        world = self.make_world({})
        world.advance(9)
        decision = world.coordinator.update(9)
        self.assertEqual(decision, CoordinatorDecision(Action.NONE, None, 0, Reason.NO_DEMAND))


# --------------------------------------------------------------------------- #
# update(): YELLOW and ALL_RED (clearance)
# --------------------------------------------------------------------------- #
class TestUpdateClearance(unittest.TestCase):
    def make_world_in_yellow(self, arrivals, **kwargs):
        world = World(arrivals, **kwargs)
        world.junction.start_phase(Phase.NORTH_SOUTH, 10)
        world.advance(10)                                       # green is over: YELLOW begins
        self.assertIs(world.junction.stage, Stage.YELLOW)
        return world

    def test_next_phase_is_queued_during_yellow_without_touching_the_signals(self):
        world = self.make_world_in_yellow({N: [0] * 5, E: [0] * 20})
        decision = world.coordinator.update(10)
        self.assertEqual(decision, CoordinatorDecision(Action.QUEUE, Phase.EAST_WEST, 40, Reason.URGENCY))
        self.assertIs(world.junction.pending_phase, Phase.EAST_WEST)
        self.assertIs(world.junction.stage, Stage.YELLOW)
        self.assertEqual(world.snapshot(), NS_YELLOW)           # still yellow, East/West still red
        self.assertTrue(world.junction.is_safe())

    def test_queued_phase_goes_green_only_after_yellow_and_all_red(self):
        world = self.make_world_in_yellow({N: [0] * 5, E: [0] * 20})
        observed = []
        for second in range(10,15):
            world.coordinator.update(second)
            observed.append(world.snapshot())
            if world.junction.stage is Stage.GREEN:
                break
            world.junction.advance()
        self.assertEqual(observed, [NS_YELLOW] * 3 + [ALL_RED] + [EW_GREEN])
        self.assertEqual(world.junction.seconds_remaining, 40)

    def test_next_phase_can_be_queued_during_all_red(self):
        world = World({N: [0] * 5, E: [0] * 20})
        world.junction.start_phase(Phase.NORTH_SOUTH, 10)
        world.advance(13)                                       # 10 green + 3 yellow -> ALL_RED
        self.assertIs(world.junction.stage, Stage.ALL_RED)
        decision = world.coordinator.update(13)
        self.assertEqual(decision.action, Action.QUEUE)
        self.assertIs(world.junction.pending_phase, Phase.EAST_WEST)
        self.assertEqual(world.snapshot(), ALL_RED)
        world.junction.advance()
        self.assertEqual(world.snapshot(), EW_GREEN)

    def test_a_newer_decision_replaces_the_queued_phase(self):
        world = self.make_world_in_yellow({N: [0] * 5, E: [0] * 20})
        world.coordinator.update(10)
        self.assertIs(world.junction.pending_phase, Phase.EAST_WEST)
        world.junction.advance()

        # Heavy North/South traffic arrives during the yellow.
        for vehicle_id in range(100, 130):
            world.lanes[N].add_vehicle(Vehicle(vehicle_id=vehicle_id, arrival_time=11))
        decision = world.coordinator.update(11)
        self.assertEqual(decision.action, Action.QUEUE)
        self.assertIs(decision.phase, Phase.NORTH_SOUTH)
        self.assertIs(world.junction.pending_phase, Phase.NORTH_SOUTH)

        for second in range(11, 14):
            world.coordinator.update(second)
            world.junction.advance()
        self.assertEqual(world.snapshot(), NS_GREEN)            # same phase again, after full clearance

    def test_same_phase_again_still_goes_through_yellow_and_all_red(self):
        world = self.make_world_in_yellow({N: [0] * 30})        # only North/South has demand
        observed = []
        for second in range(10, 15):
            world.coordinator.update(second)
            observed.append(world.snapshot())
            world.junction.advance()
        self.assertEqual(observed, [NS_YELLOW] * 3 + [ALL_RED] + [NS_GREEN])

    def test_no_demand_during_clearance_queues_nothing_and_junction_goes_idle(self):
        world = self.make_world_in_yellow({})
        decision = world.coordinator.update(10)
        self.assertEqual(decision, CoordinatorDecision(Action.NONE, None, 0, Reason.NO_DEMAND))
        self.assertIsNone(world.junction.pending_phase)
        for second in range(10, 14):
            world.coordinator.update(second)
            world.junction.advance()
        self.assertTrue(world.junction.is_idle)
        self.assertEqual(world.snapshot(), ALL_RED)

    def test_starving_phase_is_queued_next_during_clearance(self):
        world = self.make_world_in_yellow({N: [39] * 30, E: [0]}, starvation_threshold=30)
        decision = world.coordinator.update(40)
        self.assertEqual(decision.action, Action.QUEUE)
        self.assertIs(decision.phase, Phase.EAST_WEST)
        self.assertEqual(decision.reason, Reason.STARVATION)


# --------------------------------------------------------------------------- #
# Invalid inputs to update()
# --------------------------------------------------------------------------- #
class TestUpdateValidation(unittest.TestCase):
    def test_invalid_current_time_is_rejected_and_nothing_changes(self):
        world = World({N: [0] * 3})
        for bad in (-1, 1.5, "5", None, True):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    world.coordinator.update(bad)
                self.assertTrue(world.junction.is_idle)
                self.assertEqual(world.snapshot(), ALL_RED)

    def test_time_before_a_vehicles_arrival_is_rejected(self):
        world = World({N: [10]})
        with self.assertRaises(ValueError):
            world.coordinator.update(5)
        self.assertTrue(world.junction.is_idle)


# --------------------------------------------------------------------------- #
# Safety and integration with the real junction
# --------------------------------------------------------------------------- #
class TestSafetyAndIntegration(unittest.TestCase):
    def test_junction_is_safe_after_every_decision_with_static_overloaded_lanes(self):
        # Nothing ever leaves the lanes here, so both phases keep demanding green.
        world = World({N: [0] * 30, S: [0] * 20, E: [0] * 5, W: [0] * 5})
        history = []
        for second in range(400):
            world.coordinator.update(second)
            self.assertTrue(world.junction.is_safe(), msg=f"unsafe at second {second}")
            history.append(world.snapshot())
            world.junction.advance()
        check_signal_history(self, history, world.config.max_green)

    def test_both_phases_are_eventually_served_even_when_one_is_much_busier(self):
        world = World({N: [0] * 30, E: [0]})
        history = []
        for second in range(400):
            world.coordinator.update(second)
            history.append(world.snapshot())
            world.junction.advance()
        self.assertIn(NS_GREEN, history)
        self.assertIn(EW_GREEN, history)

    def test_constant_demand_never_produces_a_green_longer_than_max_green(self):
        world = World({N: [0] * 30})                            # only North/South ever has demand
        history = []
        for second in range(300):
            world.coordinator.update(second)
            history.append(world.snapshot())
            world.junction.advance()
        check_signal_history(self, history, world.config.max_green)
        self.assertGreater(history.count(NS_GREEN), 100)        # it did keep serving them

    def test_coordinator_never_causes_an_invalid_transition_error(self):
        config = SimulationConfig(duration=1500, seed=3, arrival_rate_north=0.3, arrival_rate_south=0.1,
                                  arrival_rate_east=0.2, arrival_rate_west=0.05)
        # simulate() would raise InvalidTransitionError / ValueError if the coordinator misused the junction.
        _, _, history, safe_flags, _ = simulate(config, 1500)
        self.assertTrue(all(safe_flags))
        check_signal_history(self, history, config.max_green)

    def test_mini_simulation_is_safe_and_serves_vehicles(self):
        config = SimulationConfig(duration=1200, seed=11)
        _, decisions, history, safe_flags, lanes = simulate(config, 1200)
        self.assertTrue(all(safe_flags))
        check_signal_history(self, history, config.max_green)
        self.assertGreater(sum(lane.total_departed for lane in lanes.values()), 100)
        self.assertTrue(any(d.action is Action.START for d in decisions))
        self.assertTrue(any(d.action is Action.QUEUE for d in decisions))

    def test_identical_inputs_give_identical_decisions(self):
        config = SimulationConfig(duration=800, seed=21)
        _, decisions_1, history_1, _, _ = simulate(config, 800)
        _, decisions_2, history_2, _, _ = simulate(config, 800)
        self.assertEqual(decisions_1, decisions_2)
        self.assertEqual(history_1, history_2)

    def test_light_direction_under_heavy_opposing_traffic_is_not_starved(self):
        config = SimulationConfig(duration=3000, seed=5, arrival_rate_north=0.4, arrival_rate_south=0.4,
                                  arrival_rate_east=0.03, arrival_rate_west=0.03)
        coordinator, decisions, history, safe_flags, lanes = simulate(config, 3000)
        self.assertTrue(all(safe_flags))

        # East and West vehicles do get served...
        for direction in (E, W):
            self.assertGreater(lanes[direction].total_departed, 20)

        # ...and nobody waits much longer than: threshold + one full opposing green + clearance
        # (+ a little time to discharge their own small queue).
        bound = (
            coordinator.starvation_threshold + config.max_green
            + config.yellow_duration + config.all_red_duration + 15
        )
        for direction in (E, W):
            waits = [v.waiting_time(3000) for v in lanes[direction].departed_vehicles]
            self.assertLessEqual(max(waits), bound, msg=f"{direction.name} waited too long")

        # The guard really was used in this scenario.
        self.assertTrue(any(d.reason is Reason.STARVATION for d in decisions))

    def test_coordinator_only_uses_the_junctions_public_interface(self):
        source = inspect.getsource(coordinator_module)
        self.assertIsNone(re.search(r"junction\._", source), "coordinator must not use private Junction members")


if __name__ == "__main__":
    unittest.main()
