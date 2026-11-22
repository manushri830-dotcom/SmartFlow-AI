"""
tests/test_agents.py
====================
Unit tests for agents.py (SignalAgent and PhaseRequest).

Uses only Python's built-in ``unittest``.

Run from the project folder (the one that contains config.py):
    python -m unittest tests.test_agents -v
"""

import unittest
from dataclasses import FrozenInstanceError

import agents
from agents import (
    DEFAULT_QUEUE_WEIGHT,
    DEFAULT_WAIT_WEIGHT,
    PhaseRequest,
    SignalAgent,
)
from config import Direction, SimulationConfig
from junction import Phase
from traffic import Lane, Vehicle

N, S, E, W = Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def make_lane(direction, arrival_times=()):
    """A lane of the given direction holding one vehicle per arrival time."""
    lane = Lane(direction)
    for vehicle_id, arrival_time in enumerate(arrival_times, start=1):
        lane.add_vehicle(Vehicle(vehicle_id=vehicle_id, arrival_time=arrival_time))
    return lane


def make_agent(direction=N, arrival_times=(), config=None, **weights):
    """An agent with a fresh lane. Default config: green 10-40, 0.5 vehicles/second."""
    config = config if config is not None else SimulationConfig()
    return SignalAgent(direction, make_lane(direction, arrival_times), config, **weights)


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #
class TestAgentConstruction(unittest.TestCase):
    def test_agent_stores_direction_lane_and_config(self):
        config = SimulationConfig()
        lane = make_lane(E)
        agent = SignalAgent(E, lane, config)
        self.assertIs(agent.direction, E)
        self.assertIs(agent.lane, lane)
        self.assertIs(agent.config, config)

    def test_default_weights(self):
        agent = make_agent()
        self.assertEqual(agent.queue_weight, DEFAULT_QUEUE_WEIGHT)
        self.assertEqual(agent.wait_weight, DEFAULT_WAIT_WEIGHT)

    def test_custom_weights_are_stored_as_floats(self):
        agent = make_agent(queue_weight=2, wait_weight=3)
        self.assertEqual(agent.queue_weight, 2.0)
        self.assertEqual(agent.wait_weight, 3.0)
        self.assertIsInstance(agent.queue_weight, float)

    def test_one_zero_weight_is_allowed(self):
        make_agent(queue_weight=0.0, wait_weight=1.0)
        make_agent(queue_weight=1.0, wait_weight=0.0)

    def test_invalid_direction_is_rejected(self):
        for bad in (None, "North", 0, Phase.NORTH_SOUTH):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    SignalAgent(bad, make_lane(N), SimulationConfig())

    def test_invalid_lane_is_rejected(self):
        for bad in (None, [], "lane", 5):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    SignalAgent(N, bad, SimulationConfig())

    def test_invalid_config_is_rejected(self):
        for bad in (None, {}, "config", 5):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    SignalAgent(N, make_lane(N), bad)

    def test_agent_cannot_be_given_another_directions_lane(self):
        with self.assertRaises(ValueError):
            SignalAgent(N, make_lane(S), SimulationConfig())

    def test_invalid_weights_are_rejected(self):
        bad_weights = [-1, -0.1, float("nan"), float("inf"), True, "high", None]
        for bad in bad_weights:
            with self.subTest(queue_weight=bad):
                with self.assertRaises(ValueError):
                    make_agent(queue_weight=bad)
            with self.subTest(wait_weight=bad):
                with self.assertRaises(ValueError):
                    make_agent(wait_weight=bad)

    def test_both_weights_zero_is_rejected(self):
        with self.assertRaises(ValueError):
            make_agent(queue_weight=0, wait_weight=0)


# --------------------------------------------------------------------------- #
# Direction -> phase mapping
# --------------------------------------------------------------------------- #
class TestPhaseMapping(unittest.TestCase):
    def test_each_direction_maps_to_the_correct_phase(self):
        expected = {
            N: Phase.NORTH_SOUTH,
            S: Phase.NORTH_SOUTH,
            E: Phase.EAST_WEST,
            W: Phase.EAST_WEST,
        }
        for direction, phase in expected.items():
            with self.subTest(direction=direction):
                self.assertIs(make_agent(direction).phase, phase)

    def test_north_and_south_share_a_phase_east_and_west_share_the_other(self):
        self.assertIs(make_agent(N).phase, make_agent(S).phase)
        self.assertIs(make_agent(E).phase, make_agent(W).phase)
        self.assertIsNot(make_agent(N).phase, make_agent(E).phase)

    def test_request_carries_the_agents_phase(self):
        for direction in Direction:
            with self.subTest(direction=direction):
                agent = make_agent(direction)
                self.assertIs(agent.make_request(0).phase, Phase.for_direction(direction))


# --------------------------------------------------------------------------- #
# Perception: queue length and maximum waiting time
# --------------------------------------------------------------------------- #
class TestObservation(unittest.TestCase):
    def test_queue_length_of_empty_lane_is_zero(self):
        self.assertEqual(make_agent().observe_queue_length(), 0)

    def test_queue_length_follows_the_lane(self):
        agent = make_agent(arrival_times=[0, 1, 2])
        self.assertEqual(agent.observe_queue_length(), 3)
        agent.lane.remove_vehicle(current_time=5)
        self.assertEqual(agent.observe_queue_length(), 2)
        agent.lane.add_vehicle(Vehicle(vehicle_id=99, arrival_time=6))
        self.assertEqual(agent.observe_queue_length(), 3)

    def test_max_wait_of_empty_lane_is_zero(self):
        self.assertEqual(make_agent().observe_max_wait(current_time=50), 0)

    def test_max_wait_with_a_single_vehicle(self):
        agent = make_agent(arrival_times=[4])
        self.assertEqual(agent.observe_max_wait(current_time=4), 0)
        self.assertEqual(agent.observe_max_wait(current_time=15), 11)

    def test_max_wait_is_the_wait_of_the_longest_waiting_vehicle(self):
        agent = make_agent(arrival_times=[2, 5, 9])   # the vehicle that arrived at 2 waits longest
        self.assertEqual(agent.observe_max_wait(current_time=12), 10)

    def test_max_wait_moves_to_the_next_vehicle_after_one_leaves(self):
        agent = make_agent(arrival_times=[2, 5, 9])
        agent.lane.remove_vehicle(current_time=12)
        self.assertEqual(agent.observe_max_wait(current_time=12), 7)   # now the one from time 5

    def test_max_wait_grows_as_time_passes(self):
        agent = make_agent(arrival_times=[0])
        self.assertEqual(agent.observe_max_wait(10), 10)
        self.assertEqual(agent.observe_max_wait(25), 25)

    def test_invalid_current_time_is_rejected(self):
        agent = make_agent(arrival_times=[0])
        for bad in (-1, 2.5, "10", None, True):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    agent.observe_max_wait(bad)

    def test_current_time_before_first_arrival_is_rejected(self):
        agent = make_agent(arrival_times=[10])
        with self.assertRaises(ValueError):
            agent.observe_max_wait(current_time=5)

    def test_observing_does_not_change_the_lane(self):
        agent = make_agent(arrival_times=[0, 1, 2])
        agent.observe_queue_length()
        agent.observe_max_wait(10)
        agent.make_request(10)
        self.assertEqual(agent.lane.queue_length, 3)
        self.assertEqual(agent.lane.total_departed, 0)


# --------------------------------------------------------------------------- #
# Urgency
# --------------------------------------------------------------------------- #
class TestUrgency(unittest.TestCase):
    def test_default_formula(self):
        agent = make_agent()   # weights 1.0 and 0.5
        self.assertAlmostEqual(agent.calculate_urgency(queue_length=4, max_wait=10), 4 * 1.0 + 10 * 0.5)

    def test_custom_weights(self):
        agent = make_agent(queue_weight=2.0, wait_weight=3.0)
        self.assertAlmostEqual(agent.calculate_urgency(5, 7), 2.0 * 5 + 3.0 * 7)

    def test_zero_weight_ignores_that_term(self):
        only_queue = make_agent(queue_weight=1.0, wait_weight=0.0)
        only_wait = make_agent(queue_weight=0.0, wait_weight=1.0)
        self.assertAlmostEqual(only_queue.calculate_urgency(6, 100), 6.0)
        self.assertAlmostEqual(only_wait.calculate_urgency(6, 100), 100.0)

    def test_empty_lane_has_zero_urgency(self):
        agent = make_agent()
        self.assertEqual(agent.calculate_urgency(0, 0), 0.0)
        self.assertEqual(agent.make_request(30).urgency, 0.0)

    def test_longer_queue_means_higher_urgency(self):
        agent = make_agent()
        self.assertLess(agent.calculate_urgency(3, 5), agent.calculate_urgency(8, 5))

    def test_longer_wait_means_higher_urgency(self):
        agent = make_agent()
        self.assertLess(agent.calculate_urgency(3, 5), agent.calculate_urgency(3, 50))

    def test_urgency_is_a_float(self):
        self.assertIsInstance(make_agent().calculate_urgency(2, 2), float)

    def test_invalid_inputs_are_rejected(self):
        agent = make_agent()
        for bad in (-1, 1.5, "3", None, True):
            with self.subTest(queue_length=bad):
                with self.assertRaises(ValueError):
                    agent.calculate_urgency(bad, 0)
            with self.subTest(max_wait=bad):
                with self.assertRaises(ValueError):
                    agent.calculate_urgency(0, bad)


# --------------------------------------------------------------------------- #
# Requested green duration
# --------------------------------------------------------------------------- #
class TestGreenDuration(unittest.TestCase):
    """Default config: 0.5 vehicles/second (2 s per vehicle), green limits 10-40."""

    def test_duration_follows_queue_length_inside_the_limits(self):
        agent = make_agent()
        cases = {5: 10, 6: 12, 7: 14, 12: 24, 19: 38, 20: 40}   # queue -> seconds
        for queue, seconds in cases.items():
            with self.subTest(queue=queue):
                self.assertEqual(agent.calculate_green_duration(queue), seconds)

    def test_minimum_green_is_enforced(self):
        agent = make_agent()
        for queue in (0, 1, 3, 4):
            with self.subTest(queue=queue):
                self.assertEqual(agent.calculate_green_duration(queue), 10)

    def test_maximum_green_is_enforced(self):
        agent = make_agent()
        for queue in (21, 30, 500):
            with self.subTest(queue=queue):
                self.assertEqual(agent.calculate_green_duration(queue), 40)

    def test_duration_always_stays_within_config_limits(self):
        config = SimulationConfig(min_green=8, max_green=25, fixed_green=10)
        agent = make_agent(config=config)
        for queue in range(0, 100):
            self.assertTrue(8 <= agent.calculate_green_duration(queue) <= 25)

    def test_duration_uses_the_configured_discharge_rate(self):
        fast = make_agent(config=SimulationConfig(discharge_rate=1.0))    # 1 s per vehicle
        slow = make_agent(config=SimulationConfig(discharge_rate=0.25))   # 4 s per vehicle
        self.assertEqual(fast.calculate_green_duration(15), 15)
        self.assertEqual(slow.calculate_green_duration(5), 20)

    def test_fractional_seconds_are_rounded_up(self):
        agent = make_agent(config=SimulationConfig(discharge_rate=0.4))   # 2.5 s per vehicle
        self.assertEqual(agent.calculate_green_duration(5), 13)           # 12.5 -> 13

    def test_floating_point_noise_does_not_add_a_second(self):
        # 3 / 0.3 is 10.000000000000002 in floating point; the answer must still be 10.
        config = SimulationConfig(min_green=1, max_green=40, fixed_green=20, discharge_rate=0.3)
        agent = make_agent(config=config)
        self.assertEqual(agent.calculate_green_duration(3), 10)

    def test_duration_is_an_int(self):
        self.assertIsInstance(make_agent().calculate_green_duration(7), int)

    def test_invalid_queue_length_is_rejected(self):
        agent = make_agent()
        for bad in (-1, 2.5, "4", None, True):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    agent.calculate_green_duration(bad)


# --------------------------------------------------------------------------- #
# The full request
# --------------------------------------------------------------------------- #
class TestPhaseRequest(unittest.TestCase):
    def test_request_contains_every_field(self):
        agent = make_agent(S, arrival_times=[0, 2, 4, 6, 8, 10, 12])   # 7 vehicles
        request = agent.make_request(current_time=20)

        self.assertIsInstance(request, PhaseRequest)
        self.assertIs(request.direction, S)
        self.assertIs(request.phase, Phase.NORTH_SOUTH)
        self.assertEqual(request.queue_length, 7)
        self.assertEqual(request.max_wait, 20)                      # first vehicle arrived at 0
        self.assertAlmostEqual(request.urgency, 1.0 * 7 + 0.5 * 20)  # 17.0
        self.assertEqual(request.requested_green, 14)               # 7 vehicles * 2 s
        self.assertTrue(request.has_demand)

    def test_request_matches_the_individual_calculations(self):
        agent = make_agent(W, arrival_times=[3, 3, 5, 9], queue_weight=1.5, wait_weight=0.25)
        request = agent.make_request(current_time=30)
        self.assertEqual(request.queue_length, agent.observe_queue_length())
        self.assertEqual(request.max_wait, agent.observe_max_wait(30))
        self.assertAlmostEqual(request.urgency, agent.calculate_urgency(request.queue_length, request.max_wait))
        self.assertEqual(request.requested_green, agent.calculate_green_duration(request.queue_length))

    def test_empty_lane_request(self):
        request = make_agent(E).make_request(current_time=40)
        self.assertEqual(request.queue_length, 0)
        self.assertEqual(request.max_wait, 0)
        self.assertEqual(request.urgency, 0.0)
        self.assertEqual(request.requested_green, SimulationConfig().min_green)
        self.assertFalse(request.has_demand)

    def test_request_is_immutable(self):
        request = make_agent().make_request(0)
        with self.assertRaises(FrozenInstanceError):
            request.urgency = 999.0   # type: ignore[misc]

    def test_request_reflects_the_lane_at_the_time_it_is_made(self):
        agent = make_agent(arrival_times=[0, 0, 0])
        before = agent.make_request(10)
        agent.lane.remove_vehicle(current_time=10)
        after = agent.make_request(10)
        self.assertEqual(before.queue_length, 3)
        self.assertEqual(after.queue_length, 2)
        self.assertEqual(before.queue_length, 3)   # the old request is unchanged

    def test_invalid_time_is_rejected(self):
        agent = make_agent(arrival_times=[0])
        for bad in (-5, 1.5, "now", None):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    agent.make_request(bad)

    def test_making_a_request_does_not_change_the_lane(self):
        agent = make_agent(arrival_times=[0, 1])
        agent.make_request(10)
        agent.make_request(11)
        self.assertEqual(agent.lane.queue_length, 2)
        self.assertEqual(agent.lane.total_departed, 0)


# --------------------------------------------------------------------------- #
# Independence of agents
# --------------------------------------------------------------------------- #
class TestAgentIndependence(unittest.TestCase):
    def build_four_agents(self, config=None):
        """Four agents, each with its own lane, holding different numbers of vehicles."""
        config = config if config is not None else SimulationConfig()
        arrivals = {N: [0, 1, 2, 3, 4, 5], S: [0, 1], E: [], W: [8]}
        lanes = {d: make_lane(d, arrivals[d]) for d in Direction}
        return lanes, {d: SignalAgent(d, lanes[d], config) for d in Direction}

    def test_each_agent_has_its_own_lane_and_phase_mapping(self):
        lanes, agent_map = self.build_four_agents()
        for direction, agent in agent_map.items():
            with self.subTest(direction=direction):
                self.assertIs(agent.lane, lanes[direction])
        lane_ids = {id(agent.lane) for agent in agent_map.values()}
        self.assertEqual(len(lane_ids), 4)

    def test_each_agent_reports_only_its_own_lane(self):
        _, agent_map = self.build_four_agents()
        requests = {d: a.make_request(10) for d, a in agent_map.items()}
        self.assertEqual(requests[N].queue_length, 6)
        self.assertEqual(requests[S].queue_length, 2)
        self.assertEqual(requests[E].queue_length, 0)
        self.assertEqual(requests[W].queue_length, 1)
        self.assertEqual(requests[N].max_wait, 10)
        self.assertEqual(requests[W].max_wait, 2)
        # Different lanes give different urgencies.
        self.assertEqual(len({r.urgency for r in requests.values()}), 4)

    def test_changing_one_lane_does_not_affect_other_agents(self):
        lanes, agent_map = self.build_four_agents()
        south_before = agent_map[S].make_request(10)
        east_before = agent_map[E].make_request(10)

        # Heavy traffic suddenly arrives in the North lane.
        for vehicle_id in range(100, 120):
            lanes[N].add_vehicle(Vehicle(vehicle_id=vehicle_id, arrival_time=9))

        self.assertEqual(agent_map[S].make_request(10), south_before)
        self.assertEqual(agent_map[E].make_request(10), east_before)
        self.assertEqual(agent_map[N].make_request(10).queue_length, 26)

    def test_one_agent_making_requests_does_not_affect_another(self):
        _, agent_map = self.build_four_agents()
        baseline = agent_map[W].make_request(10)
        for _ in range(5):
            agent_map[N].make_request(10)
            agent_map[S].make_request(10)
        self.assertEqual(agent_map[W].make_request(10), baseline)

    def test_agents_can_have_different_weights(self):
        lane_a = make_lane(N, [0, 0, 0, 0])
        lane_b = make_lane(S, [0, 0, 0, 0])   # identical traffic
        config = SimulationConfig()
        queue_focused = SignalAgent(N, lane_a, config, queue_weight=5.0, wait_weight=0.0)
        wait_focused = SignalAgent(S, lane_b, config, queue_weight=0.0, wait_weight=5.0)
        self.assertAlmostEqual(queue_focused.make_request(10).urgency, 5.0 * 4)
        self.assertAlmostEqual(wait_focused.make_request(10).urgency, 5.0 * 10)

    def test_two_agents_do_not_share_settings(self):
        config = SimulationConfig()
        first = SignalAgent(N, make_lane(N), config, queue_weight=3.0)
        second = SignalAgent(S, make_lane(S), config)
        first.queue_weight = 9.0   # changing one agent...
        self.assertEqual(second.queue_weight, DEFAULT_QUEUE_WEIGHT)   # ...leaves the other alone

    def test_agents_module_has_no_access_to_the_junction_class(self):
        """Agents must never control signals, so agents.py does not even import Junction."""
        self.assertFalse(hasattr(agents, "Junction"))
        agent = make_agent()
        self.assertFalse(hasattr(agent, "junction"))


if __name__ == "__main__":
    unittest.main()
