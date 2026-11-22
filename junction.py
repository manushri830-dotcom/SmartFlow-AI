"""
junction.py
===========
The traffic-signal state machine of SmartFlow AI.

This module answers ONE question: "which signal shows which colour right now,
and is it safe?"  It does NOT decide WHICH phase should get green or for how
long. That is the job of the agents and the coordinator (later modules).
They simply tell the junction what they want, and the junction carries it out
safely.

Phases
------
Directions that can safely be green together form a *phase*:

    NORTH_SOUTH : North + South
    EAST_WEST   : East  + West

The two phases conflict with each other and must never show green (or yellow)
at the same time.

The safe sequence
-----------------
    GREEN (granted seconds) -> YELLOW -> ALL-RED -> next phase's GREEN

Time convention (important for the simulator)
---------------------------------------------
* One call to ``advance()`` = one simulation second.
* The signals you read BEFORE calling ``advance()`` are the signals shown
  DURING the current second. So the simulator's loop is:
      1. read signals / discharge vehicles for this second
      2. call junction.advance()
* A green of 20 seconds is therefore shown for exactly 20 seconds.

Example timeline for start_phase(NORTH_SOUTH, 20) with yellow=3 and all-red=1,
and EAST_WEST requested as the next phase:

    seconds  0-19 : N,S green   E,W red
    seconds 20-22 : N,S yellow  E,W red
    second  23    : all red
    seconds 24-... : E,W green  N,S red
"""

from __future__ import annotations

from enum import Enum
from numbers import Integral
from typing import Any, Dict, Optional, Tuple

from config import Direction, SimulationConfig


# --------------------------------------------------------------------------- #
# Enums and a small error class
# --------------------------------------------------------------------------- #
class SignalState(Enum):
    """The colour shown by ONE traffic signal."""

    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


class Phase(Enum):
    """A group of directions that may be green together."""

    NORTH_SOUTH = "North-South"
    EAST_WEST = "East-West"

    @property
    def directions(self) -> Tuple[Direction, Direction]:
        """The two directions that belong to this phase."""
        if self is Phase.NORTH_SOUTH:
            return (Direction.NORTH, Direction.SOUTH)
        return (Direction.EAST, Direction.WEST)

    @property
    def other(self) -> "Phase":
        """The conflicting phase (NORTH_SOUTH <-> EAST_WEST)."""
        return Phase.EAST_WEST if self is Phase.NORTH_SOUTH else Phase.NORTH_SOUTH

    @classmethod
    def for_direction(cls, direction: Direction) -> "Phase":
        """Which phase does this direction belong to?"""
        if not isinstance(direction, Direction):
            raise ValueError(f"direction must be a Direction, got {direction!r}")
        if direction in (Direction.NORTH, Direction.SOUTH):
            return cls.NORTH_SOUTH
        return cls.EAST_WEST


class Stage(Enum):
    """Where the junction currently is in the safe sequence."""

    IDLE = "idle"          # all red, nothing running, a phase may start immediately
    GREEN = "green"        # the active phase is green
    YELLOW = "yellow"      # the active phase is yellow (ending)
    ALL_RED = "all_red"    # everything red while the junction clears


class InvalidTransitionError(RuntimeError):
    """Raised when a request is not allowed in the junction's current stage."""


# --------------------------------------------------------------------------- #
# Safety rule (a plain function so it can be tested with any combination)
# --------------------------------------------------------------------------- #
def signals_are_safe(signals: Dict[Direction, SignalState]) -> bool:
    """Check a set of four signal colours against the safety rules.

    A combination is SAFE only if:
      1. all four directions are present,
      2. both directions of a phase show the same colour, and
      3. at most ONE phase is showing something other than red.
         (So N/S green with E/W green is unsafe, and so is N/S yellow
         with E/W green.)
    """
    if set(signals) != set(Direction):
        return False

    active_phases = 0
    for phase in Phase:
        colours = {signals[direction] for direction in phase.directions}
        if len(colours) > 1:                 # e.g. North green but South red
            return False
        if colours != {SignalState.RED}:     # this phase is not fully red
            active_phases += 1
    return active_phases <= 1


# --------------------------------------------------------------------------- #
# The junction
# --------------------------------------------------------------------------- #
class Junction:
    """Manages the four traffic signals and moves them through safe transitions.

    Controlling it (the "write" side):
        start_phase(phase, green_duration)  - ask for a phase to get green
        extend_green(extra_seconds)         - lengthen the current green
        advance()                           - let one second pass

    Inspecting it (the "read" side):
        signal_state(direction), signal_states(), is_green(direction),
        green_directions(), stage, active_phase, pending_phase, is_idle,
        seconds_in_stage, seconds_remaining, is_safe()
    """

    def __init__(self, config: SimulationConfig) -> None:
        if not isinstance(config, SimulationConfig):
            raise ValueError(f"config must be a SimulationConfig, got {config!r}")
        # Safety needs a real yellow and a real all-red second. (SimulationConfig
        # allows all_red_duration=0, but the junction refuses to run without it.)
        if config.yellow_duration < 1:
            raise ValueError("yellow_duration must be at least 1 second for a safe junction")
        if config.all_red_duration < 1:
            raise ValueError("all_red_duration must be at least 1 second for a safe junction")

        self.config: SimulationConfig = config

        self._stage: Stage = Stage.IDLE
        self._active_phase: Optional[Phase] = None
        self._stage_duration: int = 0       # how many seconds this stage lasts
        self._seconds_elapsed: int = 0      # seconds of this stage already used up
        self._pending: Optional[Tuple[Phase, int]] = None   # (next phase, its green seconds)
        self._signals: Dict[Direction, SignalState] = {}
        self._refresh_signals()             # initial state: everything RED

    # ------------------------------------------------------------------ #
    # Controlling the junction
    # ------------------------------------------------------------------ #
    def start_phase(self, phase: Phase, green_duration: int) -> bool:
        """Ask for ``phase`` to get ``green_duration`` seconds of green.

        * If the junction is IDLE (all red, nothing running), the green starts
          immediately and ``True`` is returned.
        * If another phase is still running, the request is stored as the NEXT
          phase and ``False`` is returned. The junction first finishes the
          current green (never cut short), then runs yellow and all-red, and
          only then turns the requested phase green.
          A newer request replaces an older, not-yet-started one.

        Either way, it is impossible to jump straight from one green to
        another: the safe sequence is always followed.

        Raises
        ------
        ValueError
            ``phase`` is not a Phase, or ``green_duration`` is not a whole
            number between ``config.min_green`` and ``config.max_green``.
        InvalidTransitionError
            The requested phase is already green. Use ``extend_green()`` instead.
        """
        if not isinstance(phase, Phase):
            raise ValueError(f"phase must be a Phase, got {phase!r}")
        green_duration = self._checked_green_duration(green_duration)

        if self._stage is Stage.IDLE:
            self._begin_green(phase, green_duration)
            return True

        if self._stage is Stage.GREEN and phase is self._active_phase:
            raise InvalidTransitionError(
                f"{phase.name} is already green; use extend_green() to lengthen it"
            )

        self._pending = (phase, green_duration)
        return False

    def extend_green(self, extra_seconds: int) -> None:
        """Add ``extra_seconds`` to the green that is currently showing.

        The total green may never exceed ``config.max_green``.

        Raises
        ------
        InvalidTransitionError
            No phase is currently green.
        ValueError
            ``extra_seconds`` is not a whole number >= 1, or the total would
            exceed ``max_green``.
        """
        if self._stage is not Stage.GREEN:
            raise InvalidTransitionError("green can only be extended while a phase is green")
        if isinstance(extra_seconds, bool) or not isinstance(extra_seconds, Integral):
            raise ValueError(f"extra_seconds must be a whole number, got {extra_seconds!r}")
        if extra_seconds < 1:
            raise ValueError(f"extra_seconds must be at least 1, got {extra_seconds!r}")

        new_total = self._stage_duration + int(extra_seconds)
        if new_total > self.config.max_green:
            raise ValueError(
                f"total green of {new_total}s would exceed max_green ({self.config.max_green}s)"
            )
        self._stage_duration = new_total

    def advance(self) -> None:
        """Let ONE simulation second pass and move to the next stage if due.

        GREEN -> YELLOW -> ALL_RED -> (pending phase's GREEN, or IDLE).
        Calling this while IDLE does nothing, so the simulator may call it
        every second without checking.
        """
        if self._stage is Stage.IDLE:
            return

        self._seconds_elapsed += 1
        if self._seconds_elapsed < self._stage_duration:
            return  # the current stage still has seconds left

        # The current stage has used up all its seconds: move on.
        if self._stage is Stage.GREEN:
            self._enter_stage(Stage.YELLOW, self.config.yellow_duration)
        elif self._stage is Stage.YELLOW:
            self._enter_stage(Stage.ALL_RED, self.config.all_red_duration)
        else:  # ALL_RED is finished: the junction is clear
            if self._pending is not None:
                next_phase, green_duration = self._pending
                self._pending = None
                self._begin_green(next_phase, green_duration)
            else:
                self._active_phase = None
                self._enter_stage(Stage.IDLE, 0)

    # ------------------------------------------------------------------ #
    # Inspecting the junction
    # ------------------------------------------------------------------ #
    @property
    def stage(self) -> Stage:
        """Current stage: IDLE, GREEN, YELLOW or ALL_RED."""
        return self._stage

    @property
    def active_phase(self) -> Optional[Phase]:
        """The phase that is green or yellow, or is being cleared during ALL_RED.
        ``None`` when the junction is IDLE."""
        return self._active_phase

    @property
    def pending_phase(self) -> Optional[Phase]:
        """The phase waiting to turn green next, or ``None``."""
        return self._pending[0] if self._pending is not None else None

    @property
    def is_idle(self) -> bool:
        """True when all signals are red and nothing is running."""
        return self._stage is Stage.IDLE

    @property
    def seconds_in_stage(self) -> int:
        """Seconds of the current stage that have already passed."""
        return self._seconds_elapsed

    @property
    def seconds_remaining(self) -> int:
        """Seconds left in the current stage (0 when IDLE)."""
        return self._stage_duration - self._seconds_elapsed

    def signal_state(self, direction: Direction) -> SignalState:
        """Colour of one direction's signal right now."""
        if not isinstance(direction, Direction):
            raise ValueError(f"direction must be a Direction, got {direction!r}")
        return self._signals[direction]

    def signal_states(self) -> Dict[Direction, SignalState]:
        """A copy of all four signal colours: ``{Direction: SignalState}``."""
        return dict(self._signals)

    def is_green(self, direction: Direction) -> bool:
        """True if this direction may discharge vehicles right now."""
        return self.signal_state(direction) is SignalState.GREEN

    def green_directions(self) -> Tuple[Direction, ...]:
        """All directions that are green right now (empty, or exactly one phase)."""
        return tuple(d for d in Direction if self._signals[d] is SignalState.GREEN)

    def is_safe(self) -> bool:
        """True if the current signal combination obeys every safety rule."""
        return signals_are_safe(self._signals)

    def __repr__(self) -> str:
        phase = self._active_phase.name if self._active_phase else "None"
        return (
            f"Junction(stage={self._stage.name}, phase={phase}, "
            f"{self._seconds_elapsed}/{self._stage_duration}s)"
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _checked_green_duration(self, green_duration: Any) -> int:
        """Validate a green duration and return it as a plain int."""
        if isinstance(green_duration, bool) or not isinstance(green_duration, Integral):
            raise ValueError(f"green_duration must be a whole number of seconds, got {green_duration!r}")
        if not (self.config.min_green <= green_duration <= self.config.max_green):
            raise ValueError(
                f"green_duration must be between min_green ({self.config.min_green}) and "
                f"max_green ({self.config.max_green}), got {green_duration!r}"
            )
        return int(green_duration)

    def _begin_green(self, phase: Phase, green_duration: int) -> None:
        """Turn ``phase`` green. Only called when the junction is clear (IDLE or end of ALL_RED)."""
        self._active_phase = phase
        self._enter_stage(Stage.GREEN, green_duration)

    def _enter_stage(self, stage: Stage, duration: int) -> None:
        """Switch to a new stage, restart its clock and update the signals."""
        self._stage = stage
        self._stage_duration = duration
        self._seconds_elapsed = 0
        self._refresh_signals()

    def _refresh_signals(self) -> None:
        """Work out the four colours from (stage, active phase) and double-check safety."""
        signals = {direction: SignalState.RED for direction in Direction}
        if self._stage in (Stage.GREEN, Stage.YELLOW) and self._active_phase is not None:
            colour = SignalState.GREEN if self._stage is Stage.GREEN else SignalState.YELLOW
            for direction in self._active_phase.directions:
                signals[direction] = colour

        # Safety net: this can only fail if the code above has a bug.
        if not signals_are_safe(signals):
            raise RuntimeError(f"unsafe signal combination produced: {signals}")
        self._signals = signals
