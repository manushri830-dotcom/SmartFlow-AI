"""Tests for simulator.py (TrafficSchedule, Simulator, results, run_comparison)."""

import unittest
from dataclasses import FrozenInstanceError, replace
from typing import Dict, List, Optional, Tuple
from unittest.mock import patch

from config import Direction, SimulationConfig
from controllers import ControllerDecision, TrafficController
from coordinator import Action
from junction import Phase, SignalState, Stage, signals_are_safe
from simulator import (
    ComparisonResult,
    ControllerType,
    SimulationResult,
    Simulator,
    StepRecord,
    TrafficSchedule,
    VehicleRecord,
    run_comparison,
)
from traffic import TrafficGenerator, Vehicle

N, S, E, W = Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST
FIXED, ADAPTIVE = ControllerType.FIXED_TIME, ControllerType.ADAPTIVE
NS_DIRECTIONS = {N, S}
EW_DIRECTIONS = {E, W}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def cfg(duration: int = 100, **changes) -> SimulationConfig:
    """The default config with a shorter duration (and any other changes)."""
    return replace(SimulationConfig(), duration=duration, **changes)


def make_schedule(config: SimulationConfig, events: Optional[Dict[int, Dict[Direction, int]]] = None) -> TrafficSchedule:
    """Build an exact, hand-made schedule: events = {second: {direction: vehicle_count}}."""
    events = events or {}
    next_id = 1
    arrivals = []
    for t in range(config.duration):
        per_direction = []
        for direction in Direction:
            count = events.get(t, {}).get(direction, 0)
            per_direction.append(tuple(range(next_id, next_id + count)))
            next_id += count
        arrivals.append(tuple(per_direction))
    return TrafficSchedule(config, tuple(arrivals))


def departure_times(result: SimulationResult, direction: Direction) -> List[Optional[int]]:
    return [v.departure_time for v in result.vehicles if v.direction is direction]


def run_sim(config, controller_type, events=None, **kwargs) -> SimulationResult:
    return Simulator(config, controller_type, make_schedule(config, events), **kwargs).run()


ALLOWED_NEXT = {
    Stage.IDLE: {Stage.IDLE, Stage.GREEN},
    Stage.GREEN: {Stage.GREEN, Stage.YELLOW},
    Stage.YELLOW: {Stage.YELLOW, Stage.ALL_RED},
    Stage.ALL_RED: {Stage.ALL_RED, Stage.GREEN, Stage.IDLE},
}


# --------------------------------------------------------------------------- #
# TrafficSchedule
# --------------------------------------------------------------------------- #
class TestTrafficSchedule(unittest.TestCase):
    def test_same_config_gives_equal_schedules(self):
        config = cfg(300)
        self.assertEqual(TrafficSchedule.from_config(config), TrafficSchedule.from_config(config))

    def test_different_seed_gives_different_schedule(self):
        a = TrafficSchedule.from_config(cfg(300, seed=1))
        b = TrafficSchedule.from_config(cfg(300, seed=2))
        self.assertNotEqual(a.arrivals, b.arrivals)

    def test_matches_the_traffic_generator(self):
        config = cfg(300)
        schedule = TrafficSchedule.from_config(config)
        generator = TrafficGenerator(config)
        for direction in Direction:
            counts = generator.arrival_counts(direction)
            self.assertEqual([schedule.arrival_count(t, direction) for t in range(300)], list(counts))
            self.assertEqual(schedule.total_arrivals(direction), int(counts.sum()))

    def test_vehicle_ids_are_unique_and_consecutive(self):
        schedule = TrafficSchedule.from_config(cfg(300))
        ids = sorted(i for t in range(300) for d in Direction for i in schedule.vehicle_ids(t, d))
        self.assertEqual(ids, list(range(1, len(ids) + 1)))
        self.assertEqual(len(ids), schedule.total_arrivals())

    def test_duration_and_zero_traffic_schedule(self):
        config = cfg(50, arrival_rate_north=0, arrival_rate_south=0, arrival_rate_east=0, arrival_rate_west=0)
        schedule = TrafficSchedule.from_config(config)
        self.assertEqual(schedule.duration, 50)
        self.assertEqual(schedule.total_arrivals(), 0)

    def test_is_frozen(self):
        schedule = make_schedule(cfg(10))
        with self.assertRaises(FrozenInstanceError):
            schedule.arrivals = ()

    def test_hand_made_schedule_lookup(self):
        schedule = make_schedule(cfg(10), {3: {N: 2, E: 1}})
        self.assertEqual(schedule.vehicle_ids(3, N), (1, 2))
        self.assertEqual(schedule.vehicle_ids(3, E), (3,))
        self.assertEqual(schedule.arrival_count(3, S), 0)
        self.assertEqual(schedule.total_arrivals(), 3)
        self.assertEqual(schedule.total_arrivals(N), 2)

    def test_lookup_rejects_bad_arguments(self):
        schedule = make_schedule(cfg(10))
        for bad_time in (-1, 10, 1.5, True, None):
            with self.subTest(time=bad_time):
                with self.assertRaises(ValueError):
                    schedule.vehicle_ids(bad_time, N)
        with self.assertRaises(ValueError):
            schedule.vehicle_ids(0, "North")
        with self.assertRaises(ValueError):
            schedule.total_arrivals("North")

    def test_rejects_invalid_construction(self):
        config = cfg(3)
        empty = ((), (), (), ())
        good = (empty, empty, empty)
        with self.assertRaises(ValueError):
            TrafficSchedule("config", good)
        with self.assertRaises(ValueError):
            TrafficSchedule(config, list(good))                       # not a tuple
        with self.assertRaises(ValueError):
            TrafficSchedule(config, good[:2])                         # wrong length
        with self.assertRaises(ValueError):
            TrafficSchedule(config, (empty, empty, ((), (), ())))     # only three directions
        with self.assertRaises(ValueError):
            TrafficSchedule(config, (empty, empty, ([], (), (), ())))  # ids not a tuple
        with self.assertRaises(ValueError):
            TrafficSchedule(config, (empty, empty, ((0,), (), (), ())))   # id below 1
        with self.assertRaises(ValueError):
            TrafficSchedule(config, (((1,), (), (), ()), empty, ((1,), (), (), ())))   # duplicate id
        with self.assertRaises(ValueError):
            TrafficSchedule.from_config("config")


# --------------------------------------------------------------------------- #
# Initialisation and invalid input
# --------------------------------------------------------------------------- #
class TestInitialisation(unittest.TestCase):
    def test_initial_state(self):
        for controller_type in (FIXED, ADAPTIVE):
            with self.subTest(controller=controller_type):
                sim = Simulator(cfg(50), controller_type)
                self.assertEqual(sim.time, 0)
                self.assertFalse(sim.is_finished)
                self.assertTrue(sim.junction.is_idle)
                self.assertEqual(set(sim.lanes), set(Direction))
                self.assertTrue(all(lane.is_empty() for lane in sim.lanes.values()))
                self.assertIsInstance(sim.controller, TrafficController)
                self.assertIs(sim.junction, sim.controller.junction)
                self.assertEqual(sim.result().history, ())

    def test_controller_matches_type(self):
        self.assertEqual(Simulator(cfg(10), FIXED).controller.name, "Fixed-Time")
        self.assertEqual(Simulator(cfg(10), ADAPTIVE).controller.name, "Adaptive Multi-Agent")

    def test_adaptive_controller_uses_the_simulators_lanes(self):
        sim = Simulator(cfg(10), ADAPTIVE)
        for direction, agent in sim.controller.coordinator.agents.items():
            self.assertIs(agent.lane, sim.lanes[direction])

    def test_schedule_is_generated_when_not_given(self):
        sim = Simulator(cfg(60), FIXED)
        self.assertEqual(sim.schedule, TrafficSchedule.from_config(cfg(60)))

    def test_given_schedule_is_used_as_is(self):
        config = cfg(20)
        schedule = make_schedule(config)
        self.assertIs(Simulator(config, FIXED, schedule).schedule, schedule)

    def test_rejects_invalid_arguments(self):
        config = cfg(20)
        cases = {
            "config not a config": lambda: Simulator("config", FIXED),
            "config None": lambda: Simulator(None, FIXED),
            "type is a string": lambda: Simulator(config, "fixed_time"),
            "type None": lambda: Simulator(config, None),
            "schedule wrong type": lambda: Simulator(config, FIXED, schedule="schedule"),
            "schedule wrong duration": lambda: Simulator(config, FIXED, make_schedule(cfg(21))),
            "options for fixed": lambda: Simulator(config, FIXED, adaptive_options={"queue_weight": 1.0}),
            "options not a mapping": lambda: Simulator(config, ADAPTIVE, adaptive_options=[1]),
            "unknown option": lambda: Simulator(config, ADAPTIVE, adaptive_options={"speed": 3}),
            "bad threshold": lambda: Simulator(config, ADAPTIVE, adaptive_options={"starvation_threshold": 0}),
            "bad weights": lambda: Simulator(config, ADAPTIVE, adaptive_options={"queue_weight": 0, "wait_weight": 0}),
        }
        for name, build in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(ValueError):
                    build()

    def test_adaptive_options_reach_the_coordinator(self):
        sim = Simulator(cfg(20), ADAPTIVE, adaptive_options={"starvation_threshold": 55})
        self.assertEqual(sim.controller.coordinator.starvation_threshold, 55)

    def test_junction_refuses_config_without_all_red(self):
        with self.assertRaises(ValueError):
            Simulator(cfg(20, all_red_duration=0), FIXED)


# --------------------------------------------------------------------------- #
# Time progression
# --------------------------------------------------------------------------- #
class TestTimeProgression(unittest.TestCase):
    def test_step_advances_time_by_one(self):
        sim = Simulator(cfg(10), FIXED)
        for expected in range(1, 11):
            record = sim.step()
            self.assertEqual(record.time, expected - 1)
            self.assertEqual(sim.time, expected)
        self.assertTrue(sim.is_finished)

    def test_run_covers_exactly_the_duration(self):
        for controller_type in (FIXED, ADAPTIVE):
            with self.subTest(controller=controller_type):
                result = Simulator(cfg(75), controller_type).run()
                self.assertTrue(result.is_complete)
                self.assertEqual(result.steps_completed, 75)
                self.assertEqual([r.time for r in result.history], list(range(75)))

    def test_one_second_simulation(self):
        result = run_sim(cfg(1), FIXED, {0: {N: 1}})
        self.assertEqual(result.steps_completed, 1)
        self.assertEqual(result.total_arrived, 1)

    def test_step_after_finish_raises(self):
        sim = Simulator(cfg(3), FIXED)
        sim.run()
        with self.assertRaises(RuntimeError):
            sim.step()

    def test_reset_starts_over(self):
        sim = Simulator(cfg(30), FIXED, make_schedule(cfg(30), {0: {N: 3}}))
        sim.run()
        sim.reset()
        self.assertEqual(sim.time, 0)
        self.assertTrue(sim.junction.is_idle)
        self.assertEqual(sim.result().history, ())
        self.assertEqual(sim.result().total_arrived, 0)
        self.assertFalse(sim.is_finished)

    def test_partial_result_snapshot(self):
        sim = Simulator(cfg(40), FIXED, make_schedule(cfg(40), {0: {N: 2}}))
        for _ in range(5):
            sim.step()
        partial = sim.result()
        self.assertFalse(partial.is_complete)
        self.assertEqual(partial.steps_completed, 5)
        self.assertEqual(partial.total_arrived, 2)
        sim.step()
        self.assertEqual(partial.steps_completed, 5)          # the snapshot does not change later

    def test_running_twice_gives_identical_results(self):
        for controller_type in (FIXED, ADAPTIVE):
            with self.subTest(controller=controller_type):
                sim = Simulator(cfg(200), controller_type)
                self.assertEqual(sim.run(), sim.run())


# --------------------------------------------------------------------------- #
# Arrivals
# --------------------------------------------------------------------------- #
class TestArrivals(unittest.TestCase):
    def test_vehicles_arrive_in_the_right_lane_at_the_right_second(self):
        config = cfg(30)
        events = {2: {N: 2}, 5: {E: 1, W: 3}, 9: {S: 1}}
        result = run_sim(config, FIXED, events)
        arrivals = {(v.arrival_time, v.direction): 0 for v in result.vehicles}
        for v in result.vehicles:
            arrivals[(v.arrival_time, v.direction)] += 1
        self.assertEqual(arrivals, {(2, N): 2, (5, E): 1, (5, W): 3, (9, S): 1})
        for record in result.history:
            expected = events.get(record.time, {})
            for direction in Direction:
                self.assertEqual(record.arrived[direction], expected.get(direction, 0))

    def test_arrival_is_visible_to_the_controller_in_the_same_second(self):
        """A vehicle arriving at t=5 on a quiet junction gets a green at t=5 (arrive THEN decide)."""
        result = run_sim(cfg(30), ADAPTIVE, {5: {E: 1}})
        self.assertEqual(result.history[4].stage, Stage.IDLE)
        self.assertEqual(result.history[5].stage, Stage.GREEN)
        self.assertEqual(result.history[5].signals[E], SignalState.GREEN)
        self.assertEqual(result.history[5].decision.action, Action.START)

    def test_arrival_time_equals_the_arrival_second(self):
        result = run_sim(cfg(20), FIXED, {7: {W: 2}})
        self.assertTrue(all(v.arrival_time == 7 for v in result.vehicles))

    def test_vehicle_ids_come_from_the_schedule(self):
        config = cfg(20)
        schedule = make_schedule(config, {1: {N: 2, S: 1}, 4: {E: 1}})
        result = Simulator(config, FIXED, schedule).run()
        self.assertEqual(sorted(v.vehicle_id for v in result.vehicles), [1, 2, 3, 4])

    def test_lane_is_fifo(self):
        sim = Simulator(cfg(30), FIXED, make_schedule(cfg(30), {0: {E: 2}, 1: {E: 1}}))
        sim.step()
        sim.step()
        waiting_ids = [v.vehicle_id for v in sim.lanes[E].waiting_vehicles()]
        self.assertEqual(waiting_ids, [1, 2, 3])

    def test_generated_schedule_arrivals_are_all_delivered(self):
        config = cfg(400)
        schedule = TrafficSchedule.from_config(config)
        result = Simulator(config, FIXED, schedule).run()
        self.assertEqual(result.total_arrived, schedule.total_arrivals())
        for direction in Direction:
            self.assertEqual(
                sum(r.arrived[direction] for r in result.history), schedule.total_arrivals(direction)
            )


# --------------------------------------------------------------------------- #
# Departures (exact, hand-calculated timelines)
# --------------------------------------------------------------------------- #
class TestDepartures(unittest.TestCase):
    """Default timing: green 20, yellow 3, all-red 1, discharge 0.5 vehicles/second.

    Fixed-time timeline: NS green t=0..19, yellow 20-22, all-red 23, EW green 24..43.
    On a green, a lane earns 0.5 credit per second: one vehicle leaves every 2nd
    green second (the 2nd, 4th, 6th ...).
    """

    def test_fixed_time_north_departures_follow_the_discharge_rate(self):
        result = run_sim(cfg(60), FIXED, {0: {N: 5}})
        self.assertEqual(departure_times(result, N), [1, 3, 5, 7, 9])
        waits = [v.waiting_time(60) for v in result.vehicles]
        self.assertEqual(waits, [1, 3, 5, 7, 9])

    def test_fixed_time_waiting_for_the_other_phase(self):
        """East must wait for the EW green that starts at t=24: first departure at t=25."""
        result = run_sim(cfg(60), FIXED, {0: {E: 2}})
        self.assertEqual(departure_times(result, E), [25, 27])

    def test_both_directions_of_a_phase_discharge_together(self):
        result = run_sim(cfg(30), FIXED, {0: {N: 2, S: 2}})
        self.assertEqual(departure_times(result, N), [1, 3])
        self.assertEqual(departure_times(result, S), [1, 3])

    def test_adaptive_north_departures_follow_the_discharge_rate(self):
        result = run_sim(cfg(60), ADAPTIVE, {0: {N: 5}})
        self.assertEqual(departure_times(result, N), [1, 3, 5, 7, 9])

    def test_nothing_departs_during_yellow_or_all_red(self):
        """30 vehicles queue up; the green ends at t=20, so 10 leave and 20 stay through the clearance."""
        result = run_sim(cfg(30), FIXED, {0: {N: 30}})
        for t in range(20, 24):
            record = result.history[t]
            self.assertIn(record.stage, (Stage.YELLOW, Stage.ALL_RED))
            self.assertEqual(sum(record.departed.values()), 0, f"vehicle left during {record.stage} at t={t}")
            self.assertEqual(record.queue_lengths[N], 20)
        self.assertEqual(result.history[19].queue_lengths[N], 20)
        self.assertEqual(sum(1 for t in departure_times(result, N) if t is not None), 10)

    def test_nothing_departs_on_red(self):
        """North vehicles wait the whole EW green (t=24..43); none may leave then."""
        result = run_sim(cfg(60), FIXED, {22: {N: 3}})
        for t in range(22, 48):
            self.assertEqual(result.history[t].departed[N], 0)
        self.assertEqual(result.history[47].queue_lengths[N], 3)

    def test_departures_only_from_green_directions_in_long_runs(self):
        for controller_type in (FIXED, ADAPTIVE):
            with self.subTest(controller=controller_type):
                result = Simulator(cfg(1200), controller_type).run()
                for record in result.history:
                    for direction, count in record.departed.items():
                        if count:
                            self.assertEqual(record.signals[direction], SignalState.GREEN)

    def test_both_phases_never_discharge_in_the_same_second(self):
        for controller_type in (FIXED, ADAPTIVE):
            with self.subTest(controller=controller_type):
                result = Simulator(cfg(1200), controller_type).run()
                for record in result.history:
                    ns = sum(record.departed[d] for d in NS_DIRECTIONS)
                    ew = sum(record.departed[d] for d in EW_DIRECTIONS)
                    self.assertFalse(ns and ew, f"both phases discharged at t={record.time}")

    def test_departure_time_is_the_second_of_discharge(self):
        result = run_sim(cfg(30), FIXED, {0: {N: 3}})
        for record in result.history:
            if record.departed[N]:
                self.assertIn(record.time, departure_times(result, N))

    def test_lone_vehicle_on_an_empty_green_lane_waits_one_second(self):
        """Unused capacity is lost: a vehicle arriving at t=3 earns 0.5 at t=3 and leaves at t=4."""
        result = run_sim(cfg(30), FIXED, {3: {N: 1}})
        self.assertEqual(departure_times(result, N), [4])

    def test_one_vehicle_per_second_rate(self):
        config = cfg(40, discharge_rate=1.0)
        result = run_sim(config, FIXED, {0: {N: 4}})
        self.assertEqual(departure_times(result, N), [0, 1, 2, 3])

    def test_fractional_rate_above_one(self):
        """Rate 1.5: credit 1.5 -> 1 leaves (0.5 left), 2.0 -> 2 leave, 1.5 -> 1, 2.0 -> 2 ..."""
        config = cfg(40, discharge_rate=1.5)
        result = run_sim(config, FIXED, {0: {N: 20}})
        per_second = [record.departed[N] for record in result.history[:8]]
        self.assertEqual(per_second, [1, 2, 1, 2, 1, 2, 1, 2])

    def test_slow_rate_is_exact(self):
        """Rate 0.3 = 3 vehicles every 10 green seconds (exact fractions, no float drift)."""
        config = cfg(40, discharge_rate=0.3)
        result = run_sim(config, FIXED, {0: {N: 20}})
        departed_in_first_ten = sum(record.departed[N] for record in result.history[:10])
        self.assertEqual(departed_in_first_ten, 3)

    def test_departures_per_green_never_exceed_capacity(self):
        result = run_sim(cfg(100), FIXED, {0: {N: 100}})
        departed_during_first_green = sum(r.departed[N] for r in result.history[:20])
        self.assertEqual(departed_during_first_green, 10)     # 20 s * 0.5 vehicles/s


# --------------------------------------------------------------------------- #
# Queue tracking and vehicle accounting
# --------------------------------------------------------------------------- #
class TestQueueTrackingAndAccounting(unittest.TestCase):
    def test_queue_lengths_follow_arrivals_and_departures(self):
        result = run_sim(cfg(30), FIXED, {0: {N: 5}})
        # arrive 5 at t=0; leaves at t=1,3,5,7,9
        expected = {0: 5, 1: 4, 2: 4, 3: 3, 4: 3, 5: 2, 6: 2, 7: 1, 8: 1, 9: 0, 10: 0}
        for t, queue in expected.items():
            self.assertEqual(result.history[t].queue_lengths[N], queue, f"t={t}")

    def test_queue_equals_cumulative_arrived_minus_departed(self):
        for controller_type in (FIXED, ADAPTIVE):
            with self.subTest(controller=controller_type):
                result = Simulator(cfg(800), controller_type).run()
                arrived = {d: 0 for d in Direction}
                departed = {d: 0 for d in Direction}
                for record in result.history:
                    for d in Direction:
                        arrived[d] += record.arrived[d]
                        departed[d] += record.departed[d]
                        self.assertEqual(record.queue_lengths[d], arrived[d] - departed[d])

    def test_total_waiting_property(self):
        result = run_sim(cfg(20), FIXED, {0: {N: 4, E: 3}})
        self.assertEqual(result.history[0].total_waiting, 7)

    def test_vehicle_accounting_every_second(self):
        for controller_type in (FIXED, ADAPTIVE):
            with self.subTest(controller=controller_type):
                result = Simulator(cfg(800), controller_type).run()
                for record in result.history:
                    self.assertEqual(record.total_arrived, record.total_departed + record.total_waiting)
                    self.assertGreaterEqual(record.total_waiting, 0)

    def test_cumulative_totals_are_running_sums(self):
        result = Simulator(cfg(500), FIXED).run()
        arrived = departed = 0
        for record in result.history:
            arrived += sum(record.arrived.values())
            departed += sum(record.departed.values())
            self.assertEqual(record.total_arrived, arrived)
            self.assertEqual(record.total_departed, departed)

    def test_final_totals_match_vehicle_records(self):
        for controller_type in (FIXED, ADAPTIVE):
            with self.subTest(controller=controller_type):
                config = cfg(600)
                schedule = TrafficSchedule.from_config(config)
                result = Simulator(config, controller_type, schedule).run()
                self.assertEqual(result.total_arrived, schedule.total_arrivals())
                self.assertEqual(result.total_arrived, len(result.vehicles))
                self.assertEqual(result.total_arrived, result.total_departed + result.total_waiting)
                self.assertEqual(result.total_waiting, result.history[-1].total_waiting)
                self.assertEqual(result.total_departed, result.history[-1].total_departed)

    def test_final_queue_lengths_match_the_lanes(self):
        config = cfg(300)
        sim = Simulator(config, ADAPTIVE)
        result = sim.run()
        for direction, lane in sim.lanes.items():
            self.assertEqual(result.final_queue_lengths[direction], lane.queue_length)

    def test_final_queue_lengths_without_history(self):
        result = Simulator(cfg(10), FIXED).result()
        self.assertEqual(result.final_queue_lengths, {d: 0 for d in Direction})

    def test_departed_vehicles_have_consistent_times(self):
        for controller_type in (FIXED, ADAPTIVE):
            with self.subTest(controller=controller_type):
                result = Simulator(cfg(600), controller_type).run()
                for vehicle in result.vehicles:
                    if vehicle.has_departed:
                        self.assertGreaterEqual(vehicle.departure_time, vehicle.arrival_time)
                        self.assertLess(vehicle.departure_time, 600)

    def test_fifo_departure_order_per_direction(self):
        for controller_type in (FIXED, ADAPTIVE):
            with self.subTest(controller=controller_type):
                result = Simulator(cfg(600), controller_type).run()
                for direction in Direction:
                    times = [v.departure_time for v in result.vehicles if v.direction is direction]
                    left = [t for t in times if t is not None]
                    self.assertEqual(left, sorted(left))
                    # once one vehicle is still waiting, every later one is too
                    flags = [t is not None for t in times]
                    self.assertEqual(flags, sorted(flags, reverse=True))

    def test_vehicle_ids_are_unique(self):
        result = Simulator(cfg(600), FIXED).run()
        ids = [v.vehicle_id for v in result.vehicles]
        self.assertEqual(len(ids), len(set(ids)))

    def test_vehicles_still_waiting_at_the_end_are_reported(self):
        result = run_sim(cfg(10), FIXED, {0: {E: 3}})    # East never gets green within 10 s
        self.assertEqual(result.total_waiting, 3)
        for vehicle in result.vehicles:
            self.assertFalse(vehicle.has_departed)
            self.assertEqual(vehicle.waiting_time(result.duration), 10)


# --------------------------------------------------------------------------- #
# History content
# --------------------------------------------------------------------------- #
class TestHistoryContent(unittest.TestCase):
    def test_record_contains_all_four_directions(self):
        record = run_sim(cfg(5), FIXED).history[0]
        for field in (record.signals, record.queue_lengths, record.arrived, record.departed):
            self.assertEqual(set(field), set(Direction))

    def test_record_types(self):
        result = run_sim(cfg(5), FIXED)
        self.assertIsInstance(result, SimulationResult)
        for record in result.history:
            self.assertIsInstance(record, StepRecord)
            self.assertIsInstance(record.decision, ControllerDecision)
            self.assertIsInstance(record.stage, Stage)

    def test_signals_are_consistent_with_stage_and_phase(self):
        for controller_type in (FIXED, ADAPTIVE):
            with self.subTest(controller=controller_type):
                result = Simulator(cfg(600), controller_type).run()
                for record in result.history:
                    expected = {d: SignalState.RED for d in Direction}
                    if record.stage in (Stage.GREEN, Stage.YELLOW):
                        colour = SignalState.GREEN if record.stage is Stage.GREEN else SignalState.YELLOW
                        for d in record.active_phase.directions:
                            expected[d] = colour
                    self.assertEqual(record.signals, expected)

    def test_fixed_time_signal_timeline_is_recorded(self):
        result = run_sim(cfg(48), FIXED)
        stages = [(r.stage, r.active_phase) for r in result.history]
        expected = (
            [(Stage.GREEN, Phase.NORTH_SOUTH)] * 20 + [(Stage.YELLOW, Phase.NORTH_SOUTH)] * 3
            + [(Stage.ALL_RED, Phase.NORTH_SOUTH)] + [(Stage.GREEN, Phase.EAST_WEST)] * 20
            + [(Stage.YELLOW, Phase.EAST_WEST)] * 3 + [(Stage.ALL_RED, Phase.EAST_WEST)]
        )
        self.assertEqual(stages, expected)

    def test_decisions_are_recorded(self):
        result = run_sim(cfg(30), FIXED)
        self.assertEqual(result.history[0].decision.action, Action.START)
        self.assertEqual(result.history[1].decision.action, Action.QUEUE)
        self.assertEqual(result.history[2].decision.action, Action.NONE)

    def test_step_returns_the_record_that_is_stored(self):
        sim = Simulator(cfg(10), FIXED)
        record = sim.step()
        self.assertIs(record, sim.result().history[0])

    def test_result_objects_are_frozen(self):
        result = run_sim(cfg(5), FIXED)
        with self.assertRaises(FrozenInstanceError):
            result.controller_name = "x"
        with self.assertRaises(FrozenInstanceError):
            result.history[0].time = 99
        self.assertIsInstance(result.history, tuple)
        self.assertIsInstance(result.vehicles, tuple)


# --------------------------------------------------------------------------- #
# VehicleRecord
# --------------------------------------------------------------------------- #
class TestVehicleRecord(unittest.TestCase):
    def test_departed_vehicle_wait_is_fixed(self):
        record = VehicleRecord(1, N, arrival_time=4, departure_time=9)
        self.assertTrue(record.has_departed)
        self.assertEqual(record.waiting_time(9), 5)
        self.assertEqual(record.waiting_time(500), 5)

    def test_waiting_vehicle_wait_keeps_growing(self):
        record = VehicleRecord(1, N, arrival_time=4)
        self.assertFalse(record.has_departed)
        self.assertEqual(record.waiting_time(4), 0)
        self.assertEqual(record.waiting_time(10), 6)

    def test_rejects_bad_time(self):
        record = VehicleRecord(1, N, arrival_time=4)
        for bad in (-1, 1.5, True, None):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    record.waiting_time(bad)
        with self.assertRaises(ValueError):
            record.waiting_time(3)       # earlier than arrival


# --------------------------------------------------------------------------- #
# Signal safety
# --------------------------------------------------------------------------- #
class TestSafety(unittest.TestCase):
    def _heavy_config(self) -> SimulationConfig:
        return cfg(1500, arrival_rate_north=0.3, arrival_rate_south=0.1, arrival_rate_east=0.25, arrival_rate_west=0.05)

    def test_signals_are_safe_every_second(self):
        for controller_type in (FIXED, ADAPTIVE):
            for config in (cfg(1500), self._heavy_config()):
                with self.subTest(controller=controller_type, rate=config.arrival_rate_north):
                    for record in Simulator(config, controller_type).run().history:
                        self.assertTrue(signals_are_safe(record.signals), f"unsafe at t={record.time}")

    def test_conflicting_phases_are_never_green_together(self):
        for controller_type in (FIXED, ADAPTIVE):
            with self.subTest(controller=controller_type):
                for record in Simulator(self._heavy_config(), controller_type).run().history:
                    green = {d for d, s in record.signals.items() if s is SignalState.GREEN}
                    self.assertFalse(green & NS_DIRECTIONS and green & EW_DIRECTIONS)

    def test_safe_stage_sequence(self):
        """GREEN -> YELLOW -> ALL_RED -> next GREEN; never green straight to another green."""
        for controller_type in (FIXED, ADAPTIVE):
            with self.subTest(controller=controller_type):
                history = Simulator(self._heavy_config(), controller_type).run().history
                for before, after in zip(history, history[1:]):
                    self.assertIn(after.stage, ALLOWED_NEXT[before.stage])
                    if before.stage is Stage.GREEN and after.stage is Stage.GREEN:
                        self.assertIs(before.active_phase, after.active_phase)

    def test_green_durations_respect_limits(self):
        config = self._heavy_config()
        history = Simulator(config, ADAPTIVE).run().history
        length = 0
        lengths = []
        for record in history:
            if record.stage is Stage.GREEN:
                length += 1
            elif length:
                lengths.append(length)
                length = 0
        self.assertTrue(lengths)
        self.assertTrue(all(config.min_green <= n <= config.max_green for n in lengths))


# --------------------------------------------------------------------------- #
# Both controllers
# --------------------------------------------------------------------------- #
class TestBothControllers(unittest.TestCase):
    def test_fixed_time_ignores_traffic_pattern(self):
        config = cfg(200)
        quiet = run_sim(config, FIXED)
        busy = run_sim(config, FIXED, {t: {E: 1, W: 1} for t in range(0, 200, 3)})
        self.assertEqual([(r.stage, r.active_phase) for r in quiet.history],
                         [(r.stage, r.active_phase) for r in busy.history])

    def test_adaptive_reacts_to_traffic(self):
        config = cfg(120)
        east_only = run_sim(config, ADAPTIVE, {0: {E: 6}})
        self.assertEqual(east_only.history[0].signals[E], SignalState.GREEN)     # fixed would show NS first
        self.assertEqual(east_only.history[0].active_phase, Phase.EAST_WEST)
        self.assertEqual(east_only.total_departed, 6)

    def test_adaptive_serves_a_lonely_vehicle_via_starvation_guard(self):
        """North/South saturated, one East vehicle: it must still be served within a bounded time."""
        config = cfg(400)
        events = {t: {N: 1, S: 1} for t in range(400)}
        events[0][E] = 1
        result = run_sim(config, ADAPTIVE, events, adaptive_options={"starvation_threshold": 60})
        east = [v for v in result.vehicles if v.direction is E]
        self.assertEqual(len(east), 1)
        self.assertTrue(east[0].has_departed)
        limit = 60 + config.max_green + config.yellow_duration + config.all_red_duration + 3
        self.assertLessEqual(east[0].waiting_time(400), limit)

    def test_each_controller_serves_vehicles_on_generated_traffic(self):
        for controller_type in (FIXED, ADAPTIVE):
            with self.subTest(controller=controller_type):
                result = Simulator(cfg(900), controller_type).run()
                self.assertGreater(result.total_departed, 0)
                self.assertGreater(result.total_arrived, 0)


# --------------------------------------------------------------------------- #
# Fair comparison: identical traffic for both controllers
# --------------------------------------------------------------------------- #
class TestFairComparison(unittest.TestCase):
    def test_same_arrivals_for_both_controllers(self):
        config = cfg(900)
        schedule = TrafficSchedule.from_config(config)
        fixed = Simulator(config, FIXED, schedule).run()
        adaptive = Simulator(config, ADAPTIVE, schedule).run()

        self.assertEqual([r.arrived for r in fixed.history], [r.arrived for r in adaptive.history])
        key = lambda v: (v.vehicle_id, v.direction, v.arrival_time)
        self.assertEqual([key(v) for v in fixed.vehicles], [key(v) for v in adaptive.vehicles])
        self.assertEqual(fixed.total_arrived, adaptive.total_arrived)
        self.assertEqual(fixed.steps_completed, adaptive.steps_completed)

    def test_run_comparison_uses_one_schedule_for_both(self):
        comparison = run_comparison(cfg(600))
        self.assertIsInstance(comparison, ComparisonResult)
        self.assertEqual(comparison.fixed_time.controller_type, FIXED)
        self.assertEqual(comparison.adaptive.controller_type, ADAPTIVE)
        self.assertEqual(comparison.fixed_time.total_arrived, comparison.schedule.total_arrivals())
        self.assertEqual(comparison.adaptive.total_arrived, comparison.schedule.total_arrivals())
        self.assertEqual([r.arrived for r in comparison.fixed_time.history],
                         [r.arrived for r in comparison.adaptive.history])

    def test_run_comparison_draws_traffic_only_once(self):
        with patch("simulator.TrafficGenerator", wraps=TrafficGenerator) as generator:
            run_comparison(cfg(300))
        self.assertEqual(generator.call_count, 1)

    def test_run_comparison_uses_a_supplied_schedule(self):
        config = cfg(50)
        schedule = make_schedule(config, {0: {N: 2}})
        comparison = run_comparison(config, schedule)
        self.assertIs(comparison.schedule, schedule)
        self.assertEqual(comparison.fixed_time.total_arrived, 2)
        self.assertEqual(comparison.adaptive.total_arrived, 2)

    def test_run_comparison_passes_adaptive_options(self):
        comparison = run_comparison(cfg(50), adaptive_options={"starvation_threshold": 30})
        self.assertTrue(comparison.adaptive.is_complete)
        with self.assertRaises(ValueError):
            run_comparison(cfg(50), adaptive_options={"starvation_threshold": 0})

    def test_run_comparison_rejects_mismatched_schedule(self):
        with self.assertRaises(ValueError):
            run_comparison(cfg(50), make_schedule(cfg(60)))

    def test_running_one_controller_does_not_change_the_schedule(self):
        config = cfg(300)
        schedule = TrafficSchedule.from_config(config)
        snapshot = schedule.arrivals
        Simulator(config, FIXED, schedule).run()
        Simulator(config, ADAPTIVE, schedule).run()
        self.assertEqual(schedule.arrivals, snapshot)

    def test_both_runs_share_the_same_timing_and_discharge_rules(self):
        config = cfg(100, discharge_rate=0.4, yellow_duration=2, all_red_duration=2, fixed_green=15)
        comparison = run_comparison(config)
        for result in (comparison.fixed_time, comparison.adaptive):
            self.assertEqual(result.config, config)
        self.assertEqual(comparison.fixed_time.config, comparison.adaptive.config)

    def test_each_run_has_its_own_fresh_vehicles(self):
        config = cfg(100)
        schedule = make_schedule(config, {0: {N: 3}})
        fixed_sim = Simulator(config, FIXED, schedule)
        adaptive_sim = Simulator(config, ADAPTIVE, schedule)
        fixed_sim.run()
        # the fixed run used up its vehicles; the adaptive simulator must still be untouched
        self.assertEqual(adaptive_sim.result().total_arrived, 0)
        self.assertTrue(all(lane.is_empty() for lane in adaptive_sim.lanes.values()))


# --------------------------------------------------------------------------- #
# Reproducibility and determinism
# --------------------------------------------------------------------------- #
class TestReproducibility(unittest.TestCase):
    def test_same_seed_same_results(self):
        for controller_type in (FIXED, ADAPTIVE):
            with self.subTest(controller=controller_type):
                first = Simulator(cfg(700), controller_type).run()
                second = Simulator(cfg(700), controller_type).run()
                self.assertEqual(first, second)

    def test_different_seed_different_traffic(self):
        a = Simulator(cfg(700, seed=1), FIXED).run()
        b = Simulator(cfg(700, seed=2), FIXED).run()
        self.assertNotEqual([r.arrived for r in a.history], [r.arrived for r in b.history])

    def test_comparison_is_deterministic(self):
        self.assertEqual(run_comparison(cfg(500)), run_comparison(cfg(500)))

    def test_step_by_step_equals_run(self):
        config = cfg(300)
        schedule = TrafficSchedule.from_config(config)
        stepped = Simulator(config, ADAPTIVE, schedule)
        while not stepped.is_finished:
            stepped.step()
        self.assertEqual(stepped.result(), Simulator(config, ADAPTIVE, schedule).run())

    def test_full_default_hour_is_reproducible(self):
        first = run_comparison(SimulationConfig())
        second = run_comparison(SimulationConfig())
        self.assertEqual(first.fixed_time.total_departed, second.fixed_time.total_departed)
        self.assertEqual(first.adaptive.total_departed, second.adaptive.total_departed)
        self.assertEqual(first.fixed_time.steps_completed, 3600)


# --------------------------------------------------------------------------- #
# Empty traffic and edge cases
# --------------------------------------------------------------------------- #
class TestEmptyTrafficAndEdgeCases(unittest.TestCase):
    def test_empty_schedule_fixed_time(self):
        result = run_sim(cfg(100), FIXED)
        self.assertEqual(result.total_arrived, 0)
        self.assertEqual(result.total_departed, 0)
        self.assertEqual(result.vehicles, ())
        self.assertTrue(all(r.total_waiting == 0 for r in result.history))
        self.assertIn(Stage.GREEN, [r.stage for r in result.history])     # still cycles, safely
        self.assertTrue(all(signals_are_safe(r.signals) for r in result.history))

    def test_empty_schedule_adaptive_wastes_no_green(self):
        result = run_sim(cfg(100), ADAPTIVE)
        self.assertEqual(result.total_arrived, 0)
        self.assertTrue(all(r.stage is Stage.IDLE for r in result.history))
        self.assertTrue(all(r.decision.action is Action.NONE for r in result.history))

    def test_zero_arrival_rates_generate_an_empty_run(self):
        config = cfg(60, arrival_rate_north=0, arrival_rate_south=0, arrival_rate_east=0, arrival_rate_west=0)
        for controller_type in (FIXED, ADAPTIVE):
            with self.subTest(controller=controller_type):
                self.assertEqual(Simulator(config, controller_type).run().total_arrived, 0)

    def test_traffic_in_only_one_direction(self):
        config = cfg(200, arrival_rate_north=0.2, arrival_rate_south=0, arrival_rate_east=0, arrival_rate_west=0)
        result = Simulator(config, ADAPTIVE).run()
        self.assertTrue(all(v.direction is N for v in result.vehicles))
        self.assertGreater(result.total_departed, 0)

    def test_vehicles_arriving_in_the_last_second(self):
        result = run_sim(cfg(10), FIXED, {9: {N: 2}})
        self.assertEqual(result.total_arrived, 2)
        self.assertEqual(result.history[9].queue_lengths[N], 2 - result.history[9].departed[N])

    def test_very_heavy_traffic_stays_safe_and_consistent(self):
        config = cfg(300)
        events = {t: {N: 3, S: 3, E: 3, W: 3} for t in range(300)}
        for controller_type in (FIXED, ADAPTIVE):
            with self.subTest(controller=controller_type):
                result = run_sim(config, controller_type, events)
                self.assertEqual(result.total_arrived, 300 * 12)
                self.assertEqual(result.total_arrived, result.total_departed + result.total_waiting)
                self.assertTrue(all(signals_are_safe(r.signals) for r in result.history))

    def test_minimum_timing_configuration(self):
        config = cfg(120, min_green=1, max_green=1, fixed_green=1, yellow_duration=1, all_red_duration=1)
        for controller_type in (FIXED, ADAPTIVE):
            with self.subTest(controller=controller_type):
                result = run_sim(config, controller_type, {t: {N: 1, E: 1} for t in range(0, 120, 2)})
                self.assertTrue(all(signals_are_safe(r.signals) for r in result.history))
                self.assertEqual(result.total_arrived, result.total_departed + result.total_waiting)

    def test_custom_timing_is_respected_by_the_fixed_controller(self):
        config = cfg(60, fixed_green=12, yellow_duration=2, all_red_duration=2)
        result = run_sim(config, FIXED)
        stages = [r.stage for r in result.history[:16]]
        self.assertEqual(stages, [Stage.GREEN] * 12 + [Stage.YELLOW] * 2 + [Stage.ALL_RED] * 2)


# --------------------------------------------------------------------------- #
# The simulator polices its controllers
# --------------------------------------------------------------------------- #
class _RemovesVehicles(TrafficController):
    name = "rogue-remover"

    def __init__(self, junction, lane):
        super().__init__(junction)
        self._lane = lane

    def update(self, current_time):
        self._lane.remove_vehicle(current_time)
        return ControllerDecision(Action.NONE, None, 0)


class _AddsVehicles(TrafficController):
    name = "rogue-adder"

    def __init__(self, junction, lane):
        super().__init__(junction)
        self._lane = lane

    def update(self, current_time):
        self._lane.add_vehicle(Vehicle(vehicle_id=999, arrival_time=current_time))
        return ControllerDecision(Action.NONE, None, 0)


class _AdvancesJunction(TrafficController):
    name = "rogue-advancer"

    def update(self, current_time):
        self.junction.start_phase(Phase.NORTH_SOUTH, 10) if self.junction.is_idle else self.junction.advance()
        return ControllerDecision(Action.NONE, None, 0)


class TestSimulatorPolicesControllers(unittest.TestCase):
    def _sim_with(self, rogue_factory, events):
        config = cfg(20)
        sim = Simulator(config, FIXED, make_schedule(config, events))
        sim._controller = rogue_factory(sim)
        return sim

    def test_controller_cannot_remove_vehicles(self):
        sim = self._sim_with(lambda s: _RemovesVehicles(s.junction, s.lanes[N]), {0: {N: 2}})
        with self.assertRaises(RuntimeError):
            sim.step()

    def test_controller_cannot_add_vehicles(self):
        sim = self._sim_with(lambda s: _AddsVehicles(s.junction, s.lanes[N]), {})
        with self.assertRaises(RuntimeError):
            sim.step()

    def test_controller_cannot_advance_the_junction(self):
        sim = self._sim_with(lambda s: _AdvancesJunction(s.junction), {})
        sim.step()                      # second 0: junction idle, controller legitimately starts a phase
        with self.assertRaises(RuntimeError):
            sim.step()                  # second 1: controller calls advance() itself


if __name__ == "__main__":
    unittest.main()