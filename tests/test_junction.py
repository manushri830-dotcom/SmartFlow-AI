"""
tests/test_junction.py
======================
Unit tests for junction.py (signal state machine and safety rules).

Uses only Python's built-in ``unittest``.

Run from the project folder (the one that contains config.py):
    python -m unittest tests.test_junction -v
"""

import unittest

from config import Direction, SimulationConfig
from junction import (
    InvalidTransitionError,
    Junction,
    Phase,
    SignalState,
    Stage,
    signals_are_safe,
)

G, Y, R = SignalState.GREEN, SignalState.YELLOW, SignalState.RED
N, S, E, W = Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST

# Snapshot order is always (North, South, East, West).
NS_GREEN = (G, G, R, R)
NS_YELLOW = (Y, Y, R, R)
EW_GREEN = (R, R, G, G)
EW_YELLOW = (R, R, Y, Y)
ALL_RED = (R, R, R, R)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def make_junction(**overrides) -> Junction:
    """Junction built from the default config (green 10-40, yellow 3, all-red 1) plus overrides."""
    return Junction(SimulationConfig(**overrides))


def snapshot(junction: Junction):
    """Current signal colours as a tuple (North, South, East, West)."""
    return tuple(junction.signal_state(d) for d in Direction)


def run(junction: Junction, seconds: int) -> None:
    """Advance the junction by a number of seconds."""
    for _ in range(seconds):
        junction.advance()


def drive_alternating(junction: Junction, seconds: int, green_duration: int):
    """Keep the two phases alternating for ``seconds`` seconds.

    Every second we record the signals shown, then advance. During each green
    we make sure the opposite phase is queued, like a controller would.
    Returns the list of snapshots (one per second).
    """
    junction.start_phase(Phase.NORTH_SOUTH, green_duration)
    history = []
    for _ in range(seconds):
        if junction.stage is Stage.GREEN and junction.pending_phase is None:
            junction.start_phase(junction.active_phase.other, green_duration)
        history.append(snapshot(junction))
        junction.advance()
    return history


# --------------------------------------------------------------------------- #
# Phase definitions: correct direction combinations
# --------------------------------------------------------------------------- #
class TestPhase(unittest.TestCase):
    def test_north_south_phase_contains_north_and_south(self):
        self.assertEqual(set(Phase.NORTH_SOUTH.directions), {N, S})

    def test_east_west_phase_contains_east_and_west(self):
        self.assertEqual(set(Phase.EAST_WEST.directions), {E, W})

    def test_phases_do_not_overlap_and_cover_every_direction(self):
        ns = set(Phase.NORTH_SOUTH.directions)
        ew = set(Phase.EAST_WEST.directions)
        self.assertEqual(ns & ew, set())
        self.assertEqual(ns | ew, set(Direction))

    def test_other_returns_the_conflicting_phase(self):
        self.assertIs(Phase.NORTH_SOUTH.other, Phase.EAST_WEST)
        self.assertIs(Phase.EAST_WEST.other, Phase.NORTH_SOUTH)

    def test_for_direction_maps_each_direction_to_its_phase(self):
        self.assertIs(Phase.for_direction(N), Phase.NORTH_SOUTH)
        self.assertIs(Phase.for_direction(S), Phase.NORTH_SOUTH)
        self.assertIs(Phase.for_direction(E), Phase.EAST_WEST)
        self.assertIs(Phase.for_direction(W), Phase.EAST_WEST)

    def test_for_direction_rejects_non_direction(self):
        with self.assertRaises(ValueError):
            Phase.for_direction("North")


# --------------------------------------------------------------------------- #
# Initial state and construction
# --------------------------------------------------------------------------- #
class TestInitialState(unittest.TestCase):
    def test_every_signal_starts_red(self):
        junction = make_junction()
        for direction in Direction:
            with self.subTest(direction=direction):
                self.assertIs(junction.signal_state(direction), R)
        self.assertEqual(snapshot(junction), ALL_RED)

    def test_initial_stage_and_phase(self):
        junction = make_junction()
        self.assertIs(junction.stage, Stage.IDLE)
        self.assertTrue(junction.is_idle)
        self.assertIsNone(junction.active_phase)
        self.assertIsNone(junction.pending_phase)
        self.assertEqual(junction.green_directions(), ())
        self.assertEqual(junction.seconds_remaining, 0)

    def test_initial_state_is_safe(self):
        self.assertTrue(make_junction().is_safe())

    def test_advancing_while_idle_changes_nothing(self):
        junction = make_junction()
        run(junction, 50)
        self.assertTrue(junction.is_idle)
        self.assertEqual(snapshot(junction), ALL_RED)

    def test_constructor_rejects_non_config(self):
        for bad in (None, {}, "config", 42):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    Junction(bad)

    def test_constructor_rejects_zero_all_red(self):
        # SimulationConfig itself allows 0, but a safe junction needs a clearance second.
        config = SimulationConfig(all_red_duration=0)
        with self.assertRaises(ValueError):
            Junction(config)


# --------------------------------------------------------------------------- #
# Starting phases
# --------------------------------------------------------------------------- #
class TestStartPhase(unittest.TestCase):
    def test_starting_north_south_turns_only_north_and_south_green(self):
        junction = make_junction()
        started_now = junction.start_phase(Phase.NORTH_SOUTH, 15)
        self.assertTrue(started_now)
        self.assertEqual(snapshot(junction), NS_GREEN)
        self.assertIs(junction.stage, Stage.GREEN)
        self.assertIs(junction.active_phase, Phase.NORTH_SOUTH)
        self.assertEqual(set(junction.green_directions()), {N, S})
        self.assertTrue(junction.is_safe())

    def test_starting_east_west_turns_only_east_and_west_green(self):
        junction = make_junction()
        started_now = junction.start_phase(Phase.EAST_WEST, 15)
        self.assertTrue(started_now)
        self.assertEqual(snapshot(junction), EW_GREEN)
        self.assertIs(junction.active_phase, Phase.EAST_WEST)
        self.assertEqual(set(junction.green_directions()), {E, W})
        self.assertTrue(junction.is_safe())

    def test_is_green_reports_each_direction(self):
        junction = make_junction()
        junction.start_phase(Phase.EAST_WEST, 10)
        self.assertFalse(junction.is_green(N))
        self.assertFalse(junction.is_green(S))
        self.assertTrue(junction.is_green(E))
        self.assertTrue(junction.is_green(W))

    def test_starting_a_phase_while_another_runs_only_queues_it(self):
        junction = make_junction()
        junction.start_phase(Phase.NORTH_SOUTH, 15)
        started_now = junction.start_phase(Phase.EAST_WEST, 15)
        self.assertFalse(started_now)
        self.assertIs(junction.pending_phase, Phase.EAST_WEST)
        # Nothing changed on the signals: North/South is still green, East/West still red.
        self.assertEqual(snapshot(junction), NS_GREEN)

    def test_requesting_the_phase_that_is_already_green_is_rejected(self):
        junction = make_junction()
        junction.start_phase(Phase.NORTH_SOUTH, 15)
        with self.assertRaises(InvalidTransitionError):
            junction.start_phase(Phase.NORTH_SOUTH, 15)
        self.assertIsNone(junction.pending_phase)

    def test_invalid_phase_values_are_rejected(self):
        for bad in (None, "NORTH_SOUTH", "North-South", 1, Direction.NORTH, SignalState.GREEN):
            with self.subTest(bad=bad):
                junction = make_junction()
                with self.assertRaises(ValueError):
                    junction.start_phase(bad, 15)
                self.assertTrue(junction.is_idle)           # nothing started
                self.assertEqual(snapshot(junction), ALL_RED)

    def test_invalid_durations_are_rejected(self):
        # Default limits are min_green=10, max_green=40.
        bad_durations = [0, -5, 9, 41, 15.5, "20", None, True, float("nan")]
        for bad in bad_durations:
            with self.subTest(duration=bad):
                junction = make_junction()
                with self.assertRaises(ValueError):
                    junction.start_phase(Phase.NORTH_SOUTH, bad)
                self.assertTrue(junction.is_idle)
                self.assertEqual(snapshot(junction), ALL_RED)

    def test_min_and_max_green_are_accepted(self):
        for duration in (10, 40):
            with self.subTest(duration=duration):
                junction = make_junction()
                junction.start_phase(Phase.NORTH_SOUTH, duration)
                self.assertEqual(junction.seconds_remaining, duration)

    def test_bad_request_while_running_does_not_disturb_the_junction(self):
        junction = make_junction()
        junction.start_phase(Phase.NORTH_SOUTH, 15)
        with self.assertRaises(ValueError):
            junction.start_phase(Phase.EAST_WEST, 99)
        self.assertIsNone(junction.pending_phase)
        self.assertEqual(snapshot(junction), NS_GREEN)


# --------------------------------------------------------------------------- #
# Green duration handling (and extension)
# --------------------------------------------------------------------------- #
class TestGreenDuration(unittest.TestCase):
    def test_green_lasts_exactly_the_requested_seconds(self):
        junction = make_junction()
        junction.start_phase(Phase.NORTH_SOUTH, 12)
        for second in range(12):
            with self.subTest(second=second):
                self.assertEqual(snapshot(junction), NS_GREEN)   # shown during this second
                junction.advance()
        self.assertEqual(snapshot(junction), NS_YELLOW)           # the 13th second is yellow

    def test_remaining_time_counts_down(self):
        junction = make_junction()
        junction.start_phase(Phase.NORTH_SOUTH, 12)
        self.assertEqual(junction.seconds_remaining, 12)
        self.assertEqual(junction.seconds_in_stage, 0)
        run(junction, 5)
        self.assertEqual(junction.seconds_remaining, 7)
        self.assertEqual(junction.seconds_in_stage, 5)

    def test_extend_green_adds_seconds(self):
        junction = make_junction()
        junction.start_phase(Phase.NORTH_SOUTH, 12)
        junction.extend_green(5)
        self.assertEqual(junction.seconds_remaining, 17)
        run(junction, 16)
        self.assertEqual(snapshot(junction), NS_GREEN)   # still green after 16 s
        run(junction, 1)
        self.assertEqual(snapshot(junction), NS_YELLOW)  # yellow after 17 s

    def test_extend_green_cannot_exceed_max_green(self):
        junction = make_junction()                        # max_green = 40
        junction.start_phase(Phase.NORTH_SOUTH, 30)
        with self.assertRaises(ValueError):
            junction.extend_green(11)                     # would be 41
        self.assertEqual(junction.seconds_remaining, 30)  # unchanged after the failure
        junction.extend_green(10)                         # exactly 40 is fine
        self.assertEqual(junction.seconds_remaining, 40)

    def test_extend_green_rejects_invalid_amounts(self):
        for bad in (0, -1, 2.5, "3", None, True):
            with self.subTest(extra=bad):
                junction = make_junction()
                junction.start_phase(Phase.NORTH_SOUTH, 12)
                with self.assertRaises(ValueError):
                    junction.extend_green(bad)
                self.assertEqual(junction.seconds_remaining, 12)

    def test_extend_green_only_allowed_during_green(self):
        # IDLE
        junction = make_junction()
        with self.assertRaises(InvalidTransitionError):
            junction.extend_green(5)
        # YELLOW
        junction.start_phase(Phase.NORTH_SOUTH, 10)
        run(junction, 10)
        self.assertIs(junction.stage, Stage.YELLOW)
        with self.assertRaises(InvalidTransitionError):
            junction.extend_green(5)
        # ALL_RED
        run(junction, 3)
        self.assertIs(junction.stage, Stage.ALL_RED)
        with self.assertRaises(InvalidTransitionError):
            junction.extend_green(5)


# --------------------------------------------------------------------------- #
# Individual transitions
# --------------------------------------------------------------------------- #
class TestTransitions(unittest.TestCase):
    def test_green_to_yellow(self):
        junction = make_junction()
        junction.start_phase(Phase.NORTH_SOUTH, 10)
        run(junction, 10)
        self.assertIs(junction.stage, Stage.YELLOW)
        self.assertEqual(snapshot(junction), NS_YELLOW)
        self.assertIs(junction.active_phase, Phase.NORTH_SOUTH)

    def test_yellow_lasts_exactly_the_configured_seconds(self):
        junction = make_junction(yellow_duration=3)
        junction.start_phase(Phase.NORTH_SOUTH, 10)
        run(junction, 10)
        for second in range(3):
            with self.subTest(second=second):
                self.assertEqual(snapshot(junction), NS_YELLOW)
                junction.advance()
        self.assertEqual(snapshot(junction), ALL_RED)

    def test_yellow_to_all_red(self):
        junction = make_junction(yellow_duration=3)
        junction.start_phase(Phase.NORTH_SOUTH, 10)
        run(junction, 13)
        self.assertIs(junction.stage, Stage.ALL_RED)
        self.assertEqual(snapshot(junction), ALL_RED)
        self.assertTrue(junction.is_safe())

    def test_all_red_lasts_exactly_the_configured_seconds(self):
        junction = make_junction(all_red_duration=2)
        junction.start_phase(Phase.NORTH_SOUTH, 10)
        junction.start_phase(Phase.EAST_WEST, 10)
        run(junction, 13)                                  # 10 green + 3 yellow
        self.assertEqual(snapshot(junction), ALL_RED)      # all-red second 1
        junction.advance()
        self.assertEqual(snapshot(junction), ALL_RED)      # all-red second 2
        junction.advance()
        self.assertEqual(snapshot(junction), EW_GREEN)     # next green

    def test_all_red_to_next_green(self):
        junction = make_junction()
        junction.start_phase(Phase.NORTH_SOUTH, 10)
        junction.start_phase(Phase.EAST_WEST, 12)
        run(junction, 14)                                  # 10 + 3 + 1
        self.assertIs(junction.stage, Stage.GREEN)
        self.assertIs(junction.active_phase, Phase.EAST_WEST)
        self.assertEqual(snapshot(junction), EW_GREEN)
        self.assertEqual(junction.seconds_remaining, 12)
        self.assertIsNone(junction.pending_phase)          # the request has been used up

    def test_junction_becomes_idle_when_nothing_is_pending(self):
        junction = make_junction()
        junction.start_phase(Phase.NORTH_SOUTH, 10)
        run(junction, 14)                                  # 10 + 3 + 1
        self.assertTrue(junction.is_idle)
        self.assertIsNone(junction.active_phase)
        self.assertEqual(snapshot(junction), ALL_RED)

    def test_new_phase_can_be_started_again_once_idle(self):
        junction = make_junction()
        junction.start_phase(Phase.NORTH_SOUTH, 10)
        run(junction, 14)
        self.assertTrue(junction.start_phase(Phase.EAST_WEST, 10))
        self.assertEqual(snapshot(junction), EW_GREEN)

    def test_green_is_never_cut_short_by_a_request(self):
        junction = make_junction()
        junction.start_phase(Phase.NORTH_SOUTH, 20)
        junction.start_phase(Phase.EAST_WEST, 10)          # requested right at the start
        run(junction, 19)
        self.assertEqual(snapshot(junction), NS_GREEN)     # still green until second 20
        junction.advance()
        self.assertEqual(snapshot(junction), NS_YELLOW)

    def test_request_during_yellow_is_used_after_clearance(self):
        junction = make_junction()
        junction.start_phase(Phase.NORTH_SOUTH, 10)
        run(junction, 10)
        self.assertIs(junction.stage, Stage.YELLOW)
        self.assertFalse(junction.start_phase(Phase.EAST_WEST, 10))
        run(junction, 3)
        self.assertEqual(snapshot(junction), ALL_RED)      # clearance still happens
        run(junction, 1)
        self.assertEqual(snapshot(junction), EW_GREEN)

    def test_request_during_all_red_is_used_after_clearance(self):
        junction = make_junction()
        junction.start_phase(Phase.NORTH_SOUTH, 10)
        run(junction, 13)
        self.assertIs(junction.stage, Stage.ALL_RED)
        self.assertFalse(junction.start_phase(Phase.EAST_WEST, 10))
        run(junction, 1)
        self.assertEqual(snapshot(junction), EW_GREEN)

    def test_same_phase_may_be_requested_again_only_via_full_clearance(self):
        junction = make_junction()
        junction.start_phase(Phase.NORTH_SOUTH, 10)
        run(junction, 10)                                  # now YELLOW
        self.assertFalse(junction.start_phase(Phase.NORTH_SOUTH, 10))
        run(junction, 3)
        self.assertEqual(snapshot(junction), ALL_RED)      # clearance happened first
        run(junction, 1)
        self.assertEqual(snapshot(junction), NS_GREEN)

    def test_a_newer_request_replaces_an_older_pending_one(self):
        junction = make_junction()
        junction.start_phase(Phase.NORTH_SOUTH, 10)
        run(junction, 10)                                  # YELLOW
        junction.start_phase(Phase.NORTH_SOUTH, 10)
        junction.start_phase(Phase.EAST_WEST, 15)          # replaces the line above
        self.assertIs(junction.pending_phase, Phase.EAST_WEST)
        run(junction, 4)
        self.assertEqual(snapshot(junction), EW_GREEN)
        self.assertEqual(junction.seconds_remaining, 15)

    def test_full_second_by_second_timeline(self):
        config = SimulationConfig(
            min_green=2, max_green=10, fixed_green=2, yellow_duration=2, all_red_duration=1
        )
        junction = Junction(config)
        junction.start_phase(Phase.NORTH_SOUTH, 3)
        junction.start_phase(Phase.EAST_WEST, 2)

        observed = []
        for _ in range(11):
            observed.append(snapshot(junction))
            junction.advance()

        expected = (
            [NS_GREEN] * 3 + [NS_YELLOW] * 2 + [ALL_RED] * 1     # North/South: 3 g, 2 y, 1 red
            + [EW_GREEN] * 2 + [EW_YELLOW] * 2 + [ALL_RED] * 1   # East/West:    2 g, 2 y, 1 red
        )
        self.assertEqual(observed, expected)
        self.assertTrue(junction.is_idle)


# --------------------------------------------------------------------------- #
# Accessors
# --------------------------------------------------------------------------- #
class TestAccessors(unittest.TestCase):
    def test_signal_states_returns_all_four_directions(self):
        junction = make_junction()
        junction.start_phase(Phase.NORTH_SOUTH, 10)
        self.assertEqual(junction.signal_states(), {N: G, S: G, E: R, W: R})

    def test_signal_states_returns_a_copy(self):
        junction = make_junction()
        junction.start_phase(Phase.NORTH_SOUTH, 10)
        states = junction.signal_states()
        states[E] = G                                       # try to tamper with the copy
        self.assertIs(junction.signal_state(E), R)
        self.assertTrue(junction.is_safe())

    def test_invalid_direction_is_rejected(self):
        junction = make_junction()
        for bad in ("North", None, 0, Phase.NORTH_SOUTH):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    junction.signal_state(bad)
                with self.assertRaises(ValueError):
                    junction.is_green(bad)

    def test_repr_is_readable(self):
        junction = make_junction()
        junction.start_phase(Phase.NORTH_SOUTH, 10)
        self.assertIn("GREEN", repr(junction))
        self.assertIn("NORTH_SOUTH", repr(junction))


# --------------------------------------------------------------------------- #
# Safety rules
# --------------------------------------------------------------------------- #
class TestSafetyRule(unittest.TestCase):
    """Tests of the standalone signals_are_safe() function with hand-made signal sets."""

    @staticmethod
    def signals(north, south, east, west):
        return {N: north, S: south, E: east, W: west}

    def test_safe_combinations(self):
        safe_cases = {
            "all red": (R, R, R, R),
            "north/south green": (G, G, R, R),
            "east/west green": (R, R, G, G),
            "north/south yellow": (Y, Y, R, R),
            "east/west yellow": (R, R, Y, Y),
        }
        for name, colours in safe_cases.items():
            with self.subTest(name):
                self.assertTrue(signals_are_safe(self.signals(*colours)))

    def test_unsafe_combinations(self):
        unsafe_cases = {
            "both phases green": (G, G, G, G),
            "north and east green": (G, R, G, R),
            "south and west green": (R, G, R, G),
            "north/south green with east/west yellow": (G, G, Y, Y),
            "north/south yellow with east/west green": (Y, Y, G, G),
            "both phases yellow": (Y, Y, Y, Y),
            "north green but south red": (G, R, R, R),
            "east green but west yellow": (R, R, G, Y),
        }
        for name, colours in unsafe_cases.items():
            with self.subTest(name):
                self.assertFalse(signals_are_safe(self.signals(*colours)))

    def test_missing_direction_is_unsafe(self):
        self.assertFalse(signals_are_safe({N: R, S: R, E: R}))
        self.assertFalse(signals_are_safe({}))


class TestSafetyInvariants(unittest.TestCase):
    """Run the junction through many full cycles and check the rules every second."""

    CONFIGS = {
        "defaults": {},
        "long yellow and all-red": {"yellow_duration": 4, "all_red_duration": 3},
        "short everything": {"min_green": 1, "max_green": 5, "fixed_green": 1,
                             "yellow_duration": 1, "all_red_duration": 1},
    }

    def run_cycles(self, overrides, green_duration, seconds=400):
        config = SimulationConfig(**overrides)
        junction = Junction(config)
        # Run by hand (instead of drive_alternating) so is_safe() is checked on the real object.
        junction.start_phase(Phase.NORTH_SOUTH, green_duration)
        history = []
        for _ in range(seconds):
            if junction.stage is Stage.GREEN and junction.pending_phase is None:
                junction.start_phase(junction.active_phase.other, green_duration)
            self.assertTrue(junction.is_safe(), msg=f"unsafe at {junction!r}")
            history.append(snapshot(junction))
            junction.advance()
        return config, history

    def test_no_conflicting_signals_at_any_second(self):
        for name, overrides in self.CONFIGS.items():
            with self.subTest(name):
                green = max(SimulationConfig(**overrides).min_green, 3)
                _, history = self.run_cycles(overrides, green)
                for second, colours in enumerate(history):
                    signals = dict(zip(Direction, colours))
                    self.assertTrue(signals_are_safe(signals), msg=f"second {second}: {colours}")
                    ns_green = colours[0] is G or colours[1] is G
                    ew_green = colours[2] is G or colours[3] is G
                    self.assertFalse(ns_green and ew_green, msg=f"second {second}: {colours}")

    def test_directions_of_one_phase_always_show_the_same_colour(self):
        _, history = self.run_cycles({}, 10)
        for second, (north, south, east, west) in enumerate(history):
            self.assertIs(north, south, msg=f"second {second}")
            self.assertIs(east, west, msg=f"second {second}")

    def test_every_colour_change_follows_the_safe_sequence(self):
        """Per direction: only green->yellow, yellow->red, red->green are possible."""
        allowed = {(G, G), (Y, Y), (R, R), (G, Y), (Y, R), (R, G)}
        for name, overrides in self.CONFIGS.items():
            with self.subTest(name):
                green = max(SimulationConfig(**overrides).min_green, 3)
                _, history = self.run_cycles(overrides, green)
                for second in range(1, len(history)):
                    before, now = history[second - 1], history[second]
                    for index in range(4):
                        self.assertIn(
                            (before[index], now[index]), allowed,
                            msg=f"illegal change at second {second}: {before} -> {now}",
                        )

    def test_all_red_always_separates_conflicting_greens(self):
        for name, overrides in self.CONFIGS.items():
            with self.subTest(name):
                green = max(SimulationConfig(**overrides).min_green, 3)
                config, history = self.run_cycles(overrides, green)

                last_green_phase = None
                all_red_run = 0
                phases_seen = set()
                for second, colours in enumerate(history):
                    if colours == ALL_RED:
                        all_red_run += 1
                        continue
                    if colours[0] is G:
                        phase = Phase.NORTH_SOUTH
                    elif colours[2] is G:
                        phase = Phase.EAST_WEST
                    else:
                        all_red_run = 0
                        continue
                    phases_seen.add(phase)
                    if last_green_phase is not None and phase is not last_green_phase:
                        self.assertGreaterEqual(
                            all_red_run, config.all_red_duration,
                            msg=f"second {second}: green changed phase after only {all_red_run}s of all-red",
                        )
                    last_green_phase = phase
                    all_red_run = 0
                self.assertEqual(phases_seen, {Phase.NORTH_SOUTH, Phase.EAST_WEST})

    def test_one_complete_cycle_has_the_expected_length_and_order(self):
        """green 10 + yellow 3 + all-red 1 for each phase = 28 seconds, then it repeats."""
        junction = make_junction()
        history = drive_alternating(junction, 28, 10)
        expected = (
            [NS_GREEN] * 10 + [NS_YELLOW] * 3 + [ALL_RED]
            + [EW_GREEN] * 10 + [EW_YELLOW] * 3 + [ALL_RED]
        )
        self.assertEqual(history, expected)
        # After 28 seconds, North/South is green again.
        self.assertEqual(snapshot(junction), NS_GREEN)

    def test_repeated_advancement_stays_safe_and_consistent(self):
        junction = make_junction()
        history = drive_alternating(junction, 2000, 15)
        self.assertEqual(len(history), 2000)
        for colours in history:
            self.assertTrue(signals_are_safe(dict(zip(Direction, colours))))
        self.assertTrue(junction.is_safe())

    def test_junction_is_safe_after_every_single_advance_from_idle_to_idle(self):
        junction = make_junction()
        junction.start_phase(Phase.EAST_WEST, 10)
        junction.start_phase(Phase.NORTH_SOUTH, 10)
        steps = 0
        while not junction.is_idle:
            self.assertTrue(junction.is_safe())
            junction.advance()
            steps += 1
            self.assertLess(steps, 100, "junction never returned to idle")
        self.assertTrue(junction.is_safe())
        self.assertEqual(steps, 2 * (10 + 3 + 1))


if __name__ == "__main__":
    unittest.main()
