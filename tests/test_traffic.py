"""
tests/test_traffic.py
=====================
Unit tests for config.py and traffic.py.

Uses only Python's built-in ``unittest`` (no extra install needed).
pytest can also run this file.

Run from the project folder (the one that contains config.py):
    python -m unittest discover -s tests -t . -v
"""

import unittest
from dataclasses import FrozenInstanceError, replace

import numpy as np

from config import Direction, SimulationConfig
from traffic import Lane, TrafficGenerator, Vehicle


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def make_lane(arrival_times):
    """Build a North lane holding one vehicle per arrival time (IDs 1, 2, 3...)."""
    lane = Lane(Direction.NORTH)
    for vehicle_id, arrival_time in enumerate(arrival_times, start=1):
        lane.add_vehicle(Vehicle(vehicle_id=vehicle_id, arrival_time=arrival_time))
    return lane


def collect_all_vehicles(generator, duration):
    """Run a generator for every second and return [(id, direction, arrival_time), ...]."""
    result = []
    for t in range(duration):
        for direction, vehicles in generator.generate(t).items():
            for v in vehicles:
                result.append((v.vehicle_id, direction, v.arrival_time))
    return result


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
class TestSimulationConfig(unittest.TestCase):
    def test_default_config_is_valid(self):
        config = SimulationConfig()
        self.assertEqual(config.duration, 3600)
        self.assertEqual(config.seed, 42)
        self.assertLessEqual(config.min_green, config.fixed_green)
        self.assertLessEqual(config.fixed_green, config.max_green)

    def test_arrival_rates_cover_all_four_directions(self):
        config = SimulationConfig(
            arrival_rate_north=0.1,
            arrival_rate_south=0.2,
            arrival_rate_east=0.3,
            arrival_rate_west=0.4,
        )
        self.assertEqual(config.arrival_rate(Direction.NORTH), 0.1)
        self.assertEqual(config.arrival_rate(Direction.SOUTH), 0.2)
        self.assertEqual(config.arrival_rate(Direction.EAST), 0.3)
        self.assertEqual(config.arrival_rate(Direction.WEST), 0.4)
        self.assertEqual(set(config.arrival_rates), set(Direction))

    def test_edge_values_are_accepted(self):
        # No traffic at all, and min == max == fixed green, are legal.
        SimulationConfig(arrival_rate_north=0.0)
        SimulationConfig(min_green=15, max_green=15, fixed_green=15)
        SimulationConfig(all_red_duration=0)

    def test_config_cannot_be_changed_after_creation(self):
        config = SimulationConfig()
        with self.assertRaises(FrozenInstanceError):
            config.seed = 1  # type: ignore[misc]

    def test_invalid_configurations_are_rejected(self):
        invalid_cases = {
            "zero duration": {"duration": 0},
            "negative duration": {"duration": -5},
            "fractional duration": {"duration": 10.5},
            "boolean duration": {"duration": True},
            "negative seed": {"seed": -1},
            "negative arrival rate": {"arrival_rate_north": -0.1},
            "nan arrival rate": {"arrival_rate_east": float("nan")},
            "infinite arrival rate": {"arrival_rate_south": float("inf")},
            "text arrival rate": {"arrival_rate_west": "fast"},
            "zero min green": {"min_green": 0},
            "max green below min green": {"min_green": 20, "max_green": 10},
            "zero yellow": {"yellow_duration": 0},
            "negative all-red": {"all_red_duration": -1},
            "zero discharge rate": {"discharge_rate": 0},
            "negative discharge rate": {"discharge_rate": -0.5},
            "fixed green below min green": {"fixed_green": 5},
            "fixed green above max green": {"fixed_green": 99},
        }
        for description, overrides in invalid_cases.items():
            with self.subTest(description):
                with self.assertRaises(ValueError):
                    SimulationConfig(**overrides)

    def test_replace_revalidates(self):
        good = SimulationConfig()
        self.assertEqual(replace(good, arrival_rate_east=0.4).arrival_rate_east, 0.4)
        with self.assertRaises(ValueError):
            replace(good, arrival_rate_east=-1.0)


# --------------------------------------------------------------------------- #
# Vehicle: arrival / departure tracking and waiting time
# --------------------------------------------------------------------------- #
class TestVehicle(unittest.TestCase):
    def test_new_vehicle_is_still_waiting(self):
        v = Vehicle(vehicle_id=1, arrival_time=5)
        self.assertFalse(v.has_departed)
        self.assertIsNone(v.departure_time)

    def test_departure_is_recorded(self):
        v = Vehicle(vehicle_id=1, arrival_time=5)
        v.mark_departed(20)
        self.assertTrue(v.has_departed)
        self.assertEqual(v.departure_time, 20)

    def test_waiting_time_grows_while_in_queue(self):
        v = Vehicle(vehicle_id=1, arrival_time=10)
        self.assertEqual(v.waiting_time(10), 0)
        self.assertEqual(v.waiting_time(25), 15)

    def test_waiting_time_is_frozen_after_departure(self):
        v = Vehicle(vehicle_id=1, arrival_time=10)
        v.mark_departed(30)
        self.assertEqual(v.waiting_time(30), 20)
        self.assertEqual(v.waiting_time(500), 20)  # later clock readings change nothing

    def test_cannot_depart_before_arriving(self):
        v = Vehicle(vehicle_id=1, arrival_time=10)
        with self.assertRaises(ValueError):
            v.mark_departed(9)
        self.assertFalse(v.has_departed)  # failed attempt left it unchanged

    def test_cannot_depart_twice(self):
        v = Vehicle(vehicle_id=1, arrival_time=0)
        v.mark_departed(5)
        with self.assertRaises(ValueError):
            v.mark_departed(6)

    def test_invalid_times_are_rejected(self):
        with self.assertRaises(ValueError):
            Vehicle(vehicle_id=1, arrival_time=-1)
        with self.assertRaises(ValueError):
            Vehicle(vehicle_id=1, arrival_time=10, departure_time=5)


# --------------------------------------------------------------------------- #
# Lane: FIFO behaviour, queue length, waiting time
# --------------------------------------------------------------------------- #
class TestLane(unittest.TestCase):
    def test_vehicles_leave_in_fifo_order(self):
        lane = make_lane([0, 1, 2])
        left = [lane.remove_vehicle(current_time=10).vehicle_id for _ in range(3)]
        self.assertEqual(left, [1, 2, 3])  # first in, first out

    def test_queue_length_tracks_adds_and_removes(self):
        lane = Lane(Direction.EAST)
        self.assertEqual(lane.queue_length, 0)
        self.assertTrue(lane.is_empty())

        lane.add_vehicle(Vehicle(1, 0))
        lane.add_vehicle(Vehicle(2, 0))
        lane.add_vehicle(Vehicle(3, 1))
        self.assertEqual(lane.queue_length, 3)
        self.assertFalse(lane.is_empty())

        lane.remove_vehicle(current_time=5)
        self.assertEqual(lane.queue_length, 2)

    def test_removing_from_empty_lane_returns_none(self):
        lane = Lane(Direction.WEST)
        self.assertIsNone(lane.remove_vehicle(current_time=3))
        self.assertEqual(lane.queue_length, 0)

    def test_departure_time_is_set_when_vehicle_is_removed(self):
        lane = make_lane([4])
        vehicle = lane.remove_vehicle(current_time=12)
        self.assertEqual(vehicle.departure_time, 12)
        self.assertEqual(vehicle.waiting_time(current_time=999), 8)

    def test_lane_counts_arrivals_and_departures(self):
        lane = make_lane([0, 0, 1, 2])
        lane.remove_vehicle(5)
        lane.remove_vehicle(6)
        self.assertEqual(lane.total_arrived, 4)
        self.assertEqual(lane.total_departed, 2)
        self.assertEqual(lane.queue_length, 2)
        self.assertEqual([v.vehicle_id for v in lane.departed_vehicles], [1, 2])
        self.assertEqual([v.vehicle_id for v in lane.waiting_vehicles()], [3, 4])

    def test_first_vehicle_waiting_time(self):
        lane = make_lane([3, 7])
        self.assertEqual(lane.first_vehicle_waiting_time(current_time=10), 7)  # 10 - 3

    def test_first_vehicle_waiting_time_moves_to_next_vehicle(self):
        lane = make_lane([3, 7])
        lane.remove_vehicle(current_time=10)
        self.assertEqual(lane.first_vehicle_waiting_time(current_time=10), 3)  # 10 - 7

    def test_first_vehicle_waiting_time_is_zero_for_empty_lane(self):
        self.assertEqual(Lane(Direction.SOUTH).first_vehicle_waiting_time(50), 0)

    def test_peek_does_not_remove(self):
        lane = make_lane([0, 1])
        self.assertEqual(lane.peek().vehicle_id, 1)
        self.assertEqual(lane.queue_length, 2)

    def test_out_of_order_arrivals_are_rejected(self):
        lane = make_lane([5])
        with self.assertRaises(ValueError):
            lane.add_vehicle(Vehicle(vehicle_id=2, arrival_time=4))
        self.assertEqual(lane.queue_length, 1)

    def test_departed_vehicle_cannot_be_added(self):
        vehicle = Vehicle(vehicle_id=1, arrival_time=0)
        vehicle.mark_departed(2)
        with self.assertRaises(ValueError):
            Lane(Direction.NORTH).add_vehicle(vehicle)

    def test_failed_removal_leaves_queue_unchanged(self):
        lane = make_lane([10])
        with self.assertRaises(ValueError):
            lane.remove_vehicle(current_time=5)  # earlier than arrival
        self.assertEqual(lane.queue_length, 1)


# --------------------------------------------------------------------------- #
# TrafficGenerator: reproducibility and independence
# --------------------------------------------------------------------------- #
class TestTrafficGenerator(unittest.TestCase):
    def test_same_seed_gives_identical_traffic(self):
        config = SimulationConfig(duration=300, seed=7)
        first = TrafficGenerator(config)
        second = TrafficGenerator(config)
        for direction in Direction:
            np.testing.assert_array_equal(first.arrival_counts(direction), second.arrival_counts(direction))
        # The actual vehicle objects (IDs, directions, arrival times) match too.
        self.assertEqual(collect_all_vehicles(first, 300), collect_all_vehicles(second, 300))

    def test_different_seeds_give_different_traffic(self):
        a = TrafficGenerator(SimulationConfig(duration=500, seed=1))
        b = TrafficGenerator(SimulationConfig(duration=500, seed=2))
        self.assertFalse(np.array_equal(a.arrival_counts(Direction.NORTH), b.arrival_counts(Direction.NORTH)))

    def test_schedule_length_matches_duration(self):
        gen = TrafficGenerator(SimulationConfig(duration=123))
        for direction in Direction:
            self.assertEqual(len(gen.arrival_counts(direction)), 123)

    def test_directions_are_independent_of_each_other(self):
        base = SimulationConfig(duration=500, seed=11)
        # Same rate in every direction, yet the four streams must not be copies of each other.
        gen = TrafficGenerator(base)
        north = gen.arrival_counts(Direction.NORTH)
        for other in (Direction.SOUTH, Direction.EAST, Direction.WEST):
            self.assertFalse(np.array_equal(north, gen.arrival_counts(other)))

        # Changing only the East rate must leave North, South and West untouched.
        changed = TrafficGenerator(replace(base, arrival_rate_east=0.4))
        for direction in (Direction.NORTH, Direction.SOUTH, Direction.WEST):
            np.testing.assert_array_equal(gen.arrival_counts(direction), changed.arrival_counts(direction))
        self.assertFalse(
            np.array_equal(gen.arrival_counts(Direction.EAST), changed.arrival_counts(Direction.EAST))
        )

    def test_zero_rate_means_no_vehicles(self):
        gen = TrafficGenerator(SimulationConfig(duration=200, arrival_rate_north=0.0))
        self.assertEqual(gen.total_arrivals(Direction.NORTH), 0)

    def test_average_arrivals_match_configured_rate(self):
        config = SimulationConfig(duration=20000, seed=3, arrival_rate_north=0.2, arrival_rate_south=0.05)
        gen = TrafficGenerator(config)
        self.assertAlmostEqual(gen.total_arrivals(Direction.NORTH) / 20000, 0.2, delta=0.02)
        self.assertAlmostEqual(gen.total_arrivals(Direction.SOUTH) / 20000, 0.05, delta=0.02)

    def test_generated_vehicles_have_correct_arrival_time_and_unique_ids(self):
        config = SimulationConfig(duration=50, seed=5, arrival_rate_north=0.8, arrival_rate_west=0.8)
        gen = TrafficGenerator(config)
        all_ids = []
        for t in range(50):
            new = gen.generate(t)
            self.assertEqual(set(new), set(Direction))  # all four keys always present
            for direction, vehicles in new.items():
                self.assertEqual(len(vehicles), gen.arrival_counts(direction)[t])
                for v in vehicles:
                    self.assertEqual(v.arrival_time, t)
                    self.assertFalse(v.has_departed)
                    all_ids.append(v.vehicle_id)
        self.assertGreater(len(all_ids), 0)
        self.assertEqual(all_ids, list(range(1, len(all_ids) + 1)))  # unique, consecutive from 1

    def test_reset_replays_the_same_vehicles(self):
        gen = TrafficGenerator(SimulationConfig(duration=100, seed=9))
        first_run = collect_all_vehicles(gen, 100)
        gen.reset()
        second_run = collect_all_vehicles(gen, 100)
        self.assertEqual(first_run, second_run)

    def test_time_step_outside_simulation_is_rejected(self):
        gen = TrafficGenerator(SimulationConfig(duration=10))
        for bad_step in (-1, 10, 11):
            with self.subTest(bad_step):
                with self.assertRaises(ValueError):
                    gen.generate(bad_step)

    def test_generated_vehicles_can_be_queued_in_a_lane(self):
        """Mini integration check: generator output plugs straight into Lane."""
        gen = TrafficGenerator(SimulationConfig(duration=60, seed=21, arrival_rate_north=0.5))
        lane = Lane(Direction.NORTH)
        for t in range(60):
            for vehicle in gen.generate(t)[Direction.NORTH]:
                lane.add_vehicle(vehicle)
        self.assertEqual(lane.queue_length, gen.total_arrivals(Direction.NORTH))


if __name__ == "__main__":
    unittest.main()
