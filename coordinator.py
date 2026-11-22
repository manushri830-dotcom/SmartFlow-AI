"""
coordinator.py
==============
The central coordinator of SmartFlow AI.

Agents vs. coordinator
----------------------
* The four SignalAgents each look at ONE lane and say what they would like
  (a PhaseRequest). They never touch the lights.
* The TrafficCoordinator looks at ALL four requests, decides which phase
  should get green next, and passes that decision to the Junction through the
  Junction's public methods only (start_phase / extend_green). The Junction
  itself still guarantees safety (GREEN -> YELLOW -> ALL_RED -> GREEN), so the
  coordinator cannot make the junction unsafe even if it wanted to.

How to use it (one call per simulation second)
----------------------------------------------
    1. add this second's vehicle arrivals to the lanes
    2. decision = coordinator.update(current_time)
    3. discharge vehicles on green directions
    4. junction.advance()

Decision logic in plain words
-----------------------------
Step 1 - Collect: ask each agent for a request.
Step 2 - Group:   add up the urgencies of the two agents of each phase
                  (NORTH_SOUTH = North + South, EAST_WEST = East + West).
Step 3 - Select (see select_phase):
          a) A phase with NO waiting vehicles is never selected while the other
             phase has vehicles. If neither has vehicles, nothing is selected.
          b) STARVATION GUARD: if any direction of a phase has waited at least
             `starvation_threshold` seconds, that phase takes priority over
             normal urgency. (Details below.)
          c) Otherwise the phase with the highest combined urgency wins.
Step 4 - Act, depending on what the junction is doing right now:

    IDLE             Start the selected phase (if any phase has demand).
                     With no demand at all, do nothing: no pointless green.
    GREEN            Never interrupt. Only during the LAST second of the green,
                     if the selected phase is still the running phase, extend
                     its green (never beyond max_green).
    YELLOW / ALL_RED The green is over. Choose the next phase and queue it with
                     the junction. This is repeated every second so the final
                     choice, made in the last all-red second, uses the freshest
                     information. A newer request replaces an older one.

Tie-breaking (deterministic, no randomness, no hidden state)
------------------------------------------------------------
When two phases are equal, the phase that was NOT served last gets the turn
(`junction.active_phase.other`). If nothing was served yet (idle junction),
NORTH_SOUTH wins. This alternates fairly and gives the same answer for the same
inputs.

Starvation prevention in detail
-------------------------------
* A direction is "starving" when the longest wait in its lane is >=
  `starvation_threshold` seconds.
* A starving phase beats a non-starving phase, however urgent the other one is.
* If BOTH phases contain starving directions, the phases take turns: the phase
  that was not served last goes first (so neither can be ignored forever, even
  when the junction is badly overloaded). With nothing served yet, or if that
  does not decide it, the LONGEST wait wins, then the higher combined urgency,
  then the tie-breaking rule above.
* Two starving directions in the SAME phase (e.g. North and South) are no
  conflict: they share the phase and are served together.
* Default threshold = 2 * (max_green + yellow + all_red) = one full worst-case
  cycle of both phases (88 s with the default config). It can be overridden
  with the `starvation_threshold` argument.
* The guard only changes WHICH phase is chosen. Safety is untouched: the
  chosen phase still waits for the running green to finish and goes through
  YELLOW and ALL_RED.
* The guard cannot prevent starvation of a phase that is still being served;
  it ensures a starving phase is chosen at the next decision point, and
  max_green guarantees that decision point is never more than max_green
  seconds (plus clearance) away.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from numbers import Integral
from typing import Any, Dict, Iterable, Optional, Tuple

from agents import PhaseRequest, SignalAgent
from config import Direction, SimulationConfig
from junction import Junction, Phase, Stage


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _require_count(name: str, value: Any, minimum: int = 0) -> int:
    """Return ``value`` as an int, or raise ValueError unless it is a whole number >= minimum."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a whole number, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value!r}")
    return int(value)


def default_starvation_threshold(config: SimulationConfig) -> int:
    """Default starvation threshold in seconds: one full worst-case cycle of both phases."""
    return 2 * (config.max_green + config.yellow_duration + config.all_red_duration)


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #
class Action(Enum):
    """What the coordinator actually did to the junction this second."""

    NONE = "none"        # nothing was sent to the junction
    START = "start"      # an idle junction was given a new green phase
    QUEUE = "queue"      # the next phase was queued; it starts after yellow + all-red
    EXTEND = "extend"    # the running green was lengthened


class Reason(Enum):
    """Why a phase was selected (or why nothing was decided)."""

    NO_DEMAND = "no_demand"                  # no vehicle is waiting anywhere
    URGENCY = "urgency"                      # highest combined urgency (or tie-break)
    STARVATION = "starvation"                # a phase had a direction waiting too long
    GREEN_IN_PROGRESS = "green_in_progress"  # a green is running; nothing to decide yet


@dataclass(frozen=True)
class PhaseSummary:
    """The four requests of one phase, combined.

    Attributes
    ----------
    phase:           the phase summarised.
    urgency:         sum of the urgencies of the phase's two directions.
    max_wait:        longest wait (seconds) among the phase's two directions.
    has_demand:      True if at least one vehicle waits in either direction.
    requested_green: green seconds wanted: the largest request among directions
                     that have demand (``min_green`` if none has), kept inside
                     the junction's min/max green limits.
    """

    phase: Phase
    urgency: float
    max_wait: int
    has_demand: bool
    requested_green: int


@dataclass(frozen=True)
class CoordinatorDecision:
    """What the coordinator decided and did in one call to ``update()``.

    Attributes
    ----------
    action:         what was sent to the junction (see Action).
    phase:          the phase the coordinator selected, or None if no phase is wanted.
                    With ``Action.NONE`` this is informational only (e.g. "I
                    would like EAST_WEST next, but a green is still running").
    green_duration: seconds of green started or added (0 when nothing was sent).
    reason:         why (see Reason).
    """

    action: Action
    phase: Optional[Phase]
    green_duration: int
    reason: Reason


# --------------------------------------------------------------------------- #
# The coordinator
# --------------------------------------------------------------------------- #
class TrafficCoordinator:
    """Chooses the next phase from the four agents' requests and instructs the junction.

    Parameters
    ----------
    junction:
        The Junction to control (through its public methods only).
    agents:
        Exactly four SignalAgents, one per direction, in any order.
    starvation_threshold:
        Seconds of waiting after which a direction gets priority. ``None``
        (default) uses ``default_starvation_threshold(junction.config)``.

    The coordinator keeps no history: every decision depends only on the
    current lanes (through the agents) and the current junction state.
    """

    def __init__(
        self,
        junction: Junction,
        agents: Iterable[SignalAgent],
        starvation_threshold: Optional[int] = None,
    ) -> None:
        if not isinstance(junction, Junction):
            raise ValueError(f"junction must be a Junction, got {junction!r}")

        self.junction: Junction = junction
        self._agents: Dict[Direction, SignalAgent] = self._build_agent_map(agents)

        if starvation_threshold is None:
            starvation_threshold = default_starvation_threshold(junction.config)
        self._starvation_threshold: int = _require_count(
            "starvation_threshold", starvation_threshold, minimum=1
        )

    @staticmethod
    def _build_agent_map(agents: Iterable[SignalAgent]) -> Dict[Direction, SignalAgent]:
        """Check that there is exactly one valid agent per direction."""
        try:
            agent_list = list(agents)
        except TypeError:
            raise ValueError(f"agents must be an iterable of SignalAgent, got {agents!r}") from None

        for agent in agent_list:
            if not isinstance(agent, SignalAgent):
                raise ValueError(f"every agent must be a SignalAgent, got {agent!r}")

        by_direction: Dict[Direction, SignalAgent] = {}
        for agent in agent_list:
            if agent.direction in by_direction:
                raise ValueError(f"duplicate agent for direction {agent.direction.name}")
            by_direction[agent.direction] = agent

        missing = [direction.name for direction in Direction if direction not in by_direction]
        if missing:
            raise ValueError(f"missing agent(s) for direction(s): {', '.join(missing)}")

        return {direction: by_direction[direction] for direction in Direction}  # fixed order

    # ------------------------------------------------------------------ #
    # Read-only information
    # ------------------------------------------------------------------ #
    @property
    def starvation_threshold(self) -> int:
        """Seconds of waiting after which a direction gets priority."""
        return self._starvation_threshold

    @property
    def agents(self) -> Dict[Direction, SignalAgent]:
        """The four agents as ``{Direction: SignalAgent}`` (a copy)."""
        return dict(self._agents)

    # ------------------------------------------------------------------ #
    # Steps 1-3: collect, group, select
    # ------------------------------------------------------------------ #
    def collect_requests(self, current_time: int) -> Dict[Direction, PhaseRequest]:
        """Ask every agent for a request. Returns ``{Direction: PhaseRequest}`` in N, S, E, W order."""
        current_time = _require_count("current_time", current_time)
        return {
            direction: agent.make_request(current_time)
            for direction, agent in self._agents.items()
        }

    def summarise_phases(self, requests: Dict[Direction, PhaseRequest]) -> Dict[Phase, PhaseSummary]:
        """Combine the requests of each phase (urgencies are added up)."""
        if set(requests) != set(Direction):
            raise ValueError("requests must contain exactly one request per direction")

        config = self.junction.config
        summaries: Dict[Phase, PhaseSummary] = {}
        for phase in Phase:
            phase_requests = [requests[direction] for direction in phase.directions]
            demanding = [request for request in phase_requests if request.has_demand]
            wanted_green = max(
                (request.requested_green for request in demanding), default=config.min_green
            )
            summaries[phase] = PhaseSummary(
                phase=phase,
                urgency=sum(request.urgency for request in phase_requests),
                max_wait=max(request.max_wait for request in phase_requests),
                has_demand=bool(demanding),
                # Clamp to the JUNCTION's limits so start_phase() can never reject it.
                requested_green=max(config.min_green, min(config.max_green, wanted_green)),
            )
        return summaries

    def select_phase(self, summaries: Dict[Phase, PhaseSummary]) -> Tuple[Optional[Phase], Reason]:
        """Pick the phase that should be green next, or ``None`` if nobody is waiting.

        Order of rules: (a) only phases with demand are candidates,
        (b) starving phases beat non-starving ones (if both starve they take
        turns, then longest wait, then urgency), (c) otherwise highest urgency.
        Remaining ties are settled by ``_tie_break_rank``.
        """
        candidates = [summary for summary in summaries.values() if summary.has_demand]
        if not candidates:
            return None, Reason.NO_DEMAND

        starving = [s for s in candidates if s.max_wait >= self._starvation_threshold]
        if starving:
            best = min(
                starving,
                key=lambda s: (
                    self._alternation_rank(s.phase),
                    -s.max_wait,
                    -round(s.urgency, 6),
                    self._tie_break_rank(s.phase),
                ),
            )
            return best.phase, Reason.STARVATION

        best = min(candidates, key=lambda s: (-round(s.urgency, 6), self._tie_break_rank(s.phase)))
        return best.phase, Reason.URGENCY

    def _alternation_rank(self, phase: Phase) -> int:
        """0 for a phase that may go first, 1 for the phase that was served last.

        Used only when both phases are starving, so that they take turns.
        If nothing has been served yet, neither phase is favoured (both 0).
        """
        last_served = self.junction.active_phase
        if last_served is None:
            return 0
        return 1 if phase is last_served else 0

    def _tie_break_rank(self, phase: Phase) -> int:
        """0 for the phase that gets the turn on a tie, 1 for the other.

        The phase that was NOT served last is preferred; with nothing served
        yet (idle junction) NORTH_SOUTH is preferred.
        """
        last_served = self.junction.active_phase
        preferred = last_served.other if last_served is not None else Phase.NORTH_SOUTH
        return 0 if phase is preferred else 1

    # ------------------------------------------------------------------ #
    # Step 4: decide and act (call once per simulation second)
    # ------------------------------------------------------------------ #
    def update(self, current_time: int) -> CoordinatorDecision:
        """Look at the lanes and the junction, decide, and instruct the junction if needed.

        Call exactly once per simulation second, before ``junction.advance()``.
        """
        current_time = _require_count("current_time", current_time)   # validate before anything else
        requests = self.collect_requests(current_time)

        stage = self.junction.stage
        if stage is Stage.IDLE:
            return self._update_idle(requests)
        if stage is Stage.GREEN:
            return self._update_green(requests)
        return self._update_clearance(requests)   # YELLOW or ALL_RED

    def _update_idle(self, requests: Dict[Direction, PhaseRequest]) -> CoordinatorDecision:
        """Idle junction: start the selected phase, or stay idle if nobody is waiting."""
        summaries = self.summarise_phases(requests)
        chosen, reason = self.select_phase(summaries)
        if chosen is None:
            return CoordinatorDecision(Action.NONE, None, 0, reason)

        green_duration = summaries[chosen].requested_green
        self.junction.start_phase(chosen, green_duration)
        return CoordinatorDecision(Action.START, chosen, green_duration, reason)

    def _update_green(self, requests: Dict[Direction, PhaseRequest]) -> CoordinatorDecision:
        """Running green: never interrupt it. Maybe extend it in its last second."""
        junction = self.junction
        running_phase = junction.active_phase

        if junction.seconds_remaining > 1:
            return CoordinatorDecision(Action.NONE, running_phase, 0, Reason.GREEN_IN_PROGRESS)

        # Last second of this green: the only moment where extending is possible.
        summaries = self.summarise_phases(requests)
        chosen, reason = self.select_phase(summaries)
        if chosen is None or chosen is not running_phase or junction.pending_phase is not None:
            # Either nothing is wanted, or another phase wins (it is queued during clearance),
            # or the junction already has a queued phase.
            return CoordinatorDecision(Action.NONE, chosen, 0, reason)

        total_green = junction.seconds_in_stage + junction.seconds_remaining
        headroom = junction.config.max_green - total_green
        if headroom < 1:
            return CoordinatorDecision(Action.NONE, chosen, 0, reason)   # already at max_green

        extra_seconds = min(headroom, summaries[chosen].requested_green)
        junction.extend_green(extra_seconds)
        return CoordinatorDecision(Action.EXTEND, chosen, extra_seconds, reason)

    def _update_clearance(self, requests: Dict[Direction, PhaseRequest]) -> CoordinatorDecision:
        """YELLOW or ALL_RED: the green is over, so queue the next phase (replacing any older request)."""
        summaries = self.summarise_phases(requests)
        chosen, reason = self.select_phase(summaries)
        if chosen is None:
            return CoordinatorDecision(Action.NONE, None, 0, reason)

        green_duration = summaries[chosen].requested_green
        self.junction.start_phase(chosen, green_duration)   # junction queues it safely
        return CoordinatorDecision(Action.QUEUE, chosen, green_duration, reason)
