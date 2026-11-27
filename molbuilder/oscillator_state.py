"""A MODE'S STATE, BUILT IN STEPS -- the starts and the steps that make the
list of chances of one harmonic oscillator, a chain of them, and the state a
chain is on an oscillator (`engines/vibration-frames.md` § 9.1 module 2,
each call defined there; `science/vibrational-averaging.md` § 2e).

MODULE  oscillator_state (L2; numpy and math, and `oscillator` -- nothing
        else of molbuilder)
ROLE    the quantum mechanics of one oscillator's state, nothing molecular.
        A snapshot is one geometry; in it a mode sits at one coordinate,
        drawn at random, its moment in its own motion random too -- so the
        chance a snapshot finds the mode with ``n`` quanta, the list ``p_n``,
        is all a state holds (science § 2e.1):

        * the starts, which set a state afresh: :func:`ground`,
          :func:`level` (``n`` quanta exactly), :func:`thermal` (a
          temperature, a mean number of quanta, or neither -- the
          temperature left open, taken where the chain is placed), and, for
          comparison, the classical bell curve, :func:`classical_thermal`;
        * the events on a chain, in the order they happen:
          ``.add_quantum()``, ``.remove_quantum()`` -- each taking a list to
          a list;
        * :meth:`Chain.reduced` -- the ground state or a level with quanta
          added or removed straight after it, that level -- the form a state
          is computed in, and its closed family found from;
        * :meth:`Chain.on`, a chain placed on an :class:`oscillator.Oscillator`
          -- an :class:`OscillatorState`: its list (``populations``), its
          mean quanta and its words.
USED-BY `distribution.of_state` (a state's distribution, by the route its
        reduced chain takes), `vibration` (a chain bound to a mode),
        `vibration_frames` (the doors' choice as a chain, a set's chain
        recorded and made again), and a person's script (`postprocess` hands
        the module on as ``states``).
TRAVELS with `vibration` in both vibration bundles, which binds a chain
        to a mode (`engines/vibration-frames.md` § 2): numpy, math and
        `oscillator`, imported two ways.
"""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# TWO WAYS, because this module travels beside the job (see the header).
try:                                        # inside molbuilder
    from .oscillator import (Oscillator, _checked_quanta,
                             _finite_non_negative, _quanta_words, _whole,
                             checked_temperature_k)
except ImportError:                         # beside a job, in a bundle
    from oscillator import (  # type: ignore[no-redef]
        Oscillator, _checked_quanta, _finite_non_negative, _quanta_words,
        _whole, checked_temperature_k)


#: WHY A QUANTUM CANNOT BE REMOVED, in the one set of words every step says
#: it in (science § 2e.2: the event's chance is <n>, and nothing is left to
#: condition on).
_HOLDS_NONE = ("a quantum cannot be removed from a state that holds none -- "
               "the ground state, or a thermal state at exactly 0 K or of no "
               "mean quanta")

#: WHY A CLASSICAL STATE TAKES NO EVENT AND HAS NO LIST, in the one set of
#: words every step and answer says it in (science § 2b.3).
_COUNTS_NO_QUANTA = "a classical state counts no quanta"

#: THE PRECISION, unless a state is given its own (``Chain.on(...,
#: precision=)``; science § 2e.4, § 2e.6): a list keeps its levels up to
#: where those it leaves out hold less than this share of the averages
#: asked -- 1e-16, what a double holds, by default.
_PRECISION = 1e-16

#: HOW EXACTLY THE CUT IS PLACED: a thermal list is built out until the
#: levels beyond what is built hold this share of the precision, so the cut
#: counts the levels it leaves out and lands on its level (science § 2e.6:
#: built only to the precision itself, a 30 cm-1 mode at 300 K with one
#: quantum added, asked to 1e-8, keeps 243 levels and leaves out 1.04 of the
#: precision; placed within 1e-4 of it, 244, leaving out 0.92).
_CUT_PLACED_WITHIN = 1e-4

#: THE HIGHEST AVERAGE A LIST IS CUT FOR, unless more is asked:
#: ``<x^12>``, the highest even average the largest rule the doors offer
#: reproduces -- seven frames, exact to degree 13 (`vibration_frames`'
#: ``OFFERED_NODES``; science § 2e.4: 227 levels for a 50 cm-1 mode at
#: 300 K).
_DEFAULT_ORDER = 12


# --------------------------------------------------------------------- #
#  A step, and a chain of them                                           #
# --------------------------------------------------------------------- #

@dataclass(frozen=True)
class Step:
    """One start or step, by ``name``, and its ``values`` -- ``(key,
    value)`` pairs, sorted by key, so two steps that are one compare equal."""
    name: str
    values: Tuple[Tuple[str, Any], ...] = ()

    def get(self, key: str, default: Any = None) -> Any:
        """The step's value of ``key``."""
        return dict(self.values).get(key, default)

    def with_values(self, **changes) -> "Step":
        """The step with ``changes`` to its values."""
        merged = dict(self.values)
        merged.update(changes)
        return Step(self.name, tuple(sorted(merged.items())))


def _step(name: str, **values) -> Step:
    return Step(name, tuple(sorted(values.items())))


#: EACH START AND THE VALUES IT TAKES (science § 2e.2), in its maker's order.
_START_KEYS = {"ground": (), "level": ("quanta",),
               "thermal": ("temperature_k", "mean_quanta"),
               "classical_thermal": ("temperature_k",)}

#: THE EVENTS a start is followed by; each takes no value.
_EVENTS = ("add_quantum", "remove_quantum")


def _checked_values(name: Any, values: Any) -> Dict[str, Any]:
    """A START'S VALUES, EACH CHECKED (science § 2e.2): ``name`` a start and
    ``values`` a mapping of what it takes, each value checked as its maker
    checks it -- answered in their one form, a value of ``None`` left out.
    Asked by :func:`_start_step`.  Refused by name: a name that is no start,
    values that are no mapping, a key the start does not take, and what
    each value's check refuses."""
    if name not in _START_KEYS:
        raise ValueError(f"a chain starts with ground, level, thermal or "
                         f"classical_thermal; its first step is {name!r}")
    if not isinstance(values, dict):
        raise ValueError(f"a start's values are a mapping of what it takes; "
                         f"{name} got {values!r}")
    takes = _START_KEYS[name]
    extra = sorted(set(values) - set(takes))
    if extra:
        raise ValueError(f"{name} takes {' or '.join(takes) or 'no value'}; "
                         f"got {', '.join(map(repr, extra))}")
    given = {k: v for k, v in values.items() if v is not None}
    if name == "level":
        return {"quanta": _whole(values.get("quanta"), "a level", 0)}
    checked = {}
    if "temperature_k" in given:
        checked["temperature_k"] = checked_temperature_k(
            given["temperature_k"])
    if "mean_quanta" in given:
        checked["mean_quanta"] = _checked_quanta(given["mean_quanta"])
    return checked


def _start_step(name: Any, values: Any) -> Step:
    """THE ONE MAKER OF A START (science § 2e.2): its values checked
    (:func:`_checked_values`) and a thermal start named one way -- by a
    temperature or by a mean, not both -- answered as the start's step.
    Asked by every start's maker, by a chain built by hand
    (:func:`_checked_steps`) and by a chain made again from its record
    (:meth:`Chain.from_record`), so a start is made the same whichever way
    it is written."""
    checked = _checked_values(name, values)
    if name == "thermal" and len(checked) == 2:
        raise ValueError("a thermal state is named by a temperature or by a "
                         "mean number of quanta, not both")
    return _step(name, **checked)


def _checked_steps(steps: Any) -> None:
    """A CHAIN'S STEPS, CHECKED (science § 2e.2): a start -- made as
    :func:`_start_step` makes it, so a thermal start names a temperature or
    a mean, not both -- and after it only the events, each of no value, a
    classical start followed by none.  Refused by name, so a chain built by
    hand is never computed as something else."""
    if (not isinstance(steps, tuple) or not steps
            or not all(isinstance(s, Step) for s in steps)):
        raise ValueError(f"a chain is a start and the events after it, each "
                         f"a Step; got {steps!r}")
    start = steps[0]
    if _start_step(start.name, dict(start.values)) != start:
        raise ValueError(f"a start's values are in their one form, as its "
                         f"maker writes them; got {start!r}")
    for s in steps[1:]:
        if s.name not in _EVENTS or s.values:
            raise ValueError(f"after its start a chain holds add_quantum and "
                             f"remove_quantum, each of no value; got {s!r}")
    if steps[0].name == "classical_thermal" and len(steps) > 1:
        raise ValueError(f"{_COUNTS_NO_QUANTA}: a quantum is added or "
                         f"removed only on a quantum start -- ground, level "
                         f"or thermal")


@dataclass(frozen=True)
class Chain:
    """A START AND THE STEPS AFTER IT, written without a frequency (science
    § 2e.2): each method returns a new chain.  A chain has no numbers until
    it is placed: ``.on(oscillator)`` makes it a state that has them.  Built
    by hand, its steps are checked as the starts and events check them
    (:func:`_checked_steps`)."""
    steps: Tuple[Step, ...]

    def __post_init__(self) -> None:
        _checked_steps(self.steps)

    @property
    def start(self) -> Step:
        return self.steps[0]

    @property
    def is_classical(self) -> bool:
        """Whether the chain starts classical -- no zero-point motion, no
        quanta to count."""
        return self.start.name == "classical_thermal"

    @property
    def temperature_open(self) -> bool:
        """Whether its start is a thermal one given neither a temperature nor
        a mean -- the temperature taken where the chain is placed."""
        s = self.start
        return (s.name in ("thermal", "classical_thermal")
                and s.get("temperature_k") is None
                and s.get("mean_quanta") is None)

    @property
    def words(self) -> str:
        """The chain in words, each step saying what it states
        (:func:`_steps_words`) -- *thermal at 300 K; one quantum added*;
        placed on an oscillator, its state's words add the mean quanta."""
        return _steps_words(self.steps)

    def words_with_source(self, source: Optional[str]) -> str:
        """Its words, its start's temperature said as ``source``'s -- the
        holder an open start took it from (*thermal at 300 K, the motion's
        temperature*); ``None`` says none (:func:`_steps_words`)."""
        return _steps_words(self.steps, source)

    def with_temperature(self, temperature_k: float) -> "Chain":
        """The chain, its open start's temperature filled -- the one filling
        :meth:`on` makes before it places the chain.  Refused by name: a
        start that takes no temperature (the ground state, a level) and one
        that names its own temperature or mean."""
        if self.start.name in ("ground", "level"):
            raise ValueError(f"{self.start.name} takes no temperature; a "
                             f"temperature fills only an open thermal start")
        if not self.temperature_open:
            raise ValueError(f"{self.start.name} names its own temperature or "
                             f"mean; a temperature fills only an open one")
        return Chain((self.start.with_values(
            temperature_k=checked_temperature_k(temperature_k)),)
            + self.steps[1:])

    def to_record(self) -> List[Dict[str, Any]]:
        """THE CHAIN AS A FRAME SET RECORDS IT (`engines/vibration-frames.md`
        § 5): its steps in order, each ``{"step": name, "values": {...}}``,
        the chain as given -- :meth:`from_record` makes it again."""
        return [{"step": s.name, "values": dict(s.values)}
                for s in self.steps]

    @classmethod
    def from_record(cls, record: Any) -> "Chain":
        """A CHAIN MADE AGAIN FROM ITS RECORD (:meth:`to_record`), by the
        starts and the events themselves, so it is checked as they check.
        Refused by name: what is no list of steps, a first step that is no
        start, a later one that is no event or holds a value, a start's
        values that are no mapping or hold a key it does not take, and what a
        start refuses."""
        steps = list(record) if isinstance(record, (list, tuple)) else []
        if not steps or not all(isinstance(s, dict) for s in steps):
            raise ValueError(f"a chain's record is its steps in order, each "
                             f"{{\"step\", \"values\"}}; got {record!r}")
        first = steps[0]
        chain = Chain((_start_step(first.get("step"),
                                   first.get("values") or {}),))
        for s in steps[1:]:
            if s.get("step") not in _EVENTS or s.get("values"):
                raise ValueError(f"after its start a chain holds add_quantum "
                                 f"and remove_quantum, each of no value; its "
                                 f"record holds {s!r}")
            chain = (chain.add_quantum() if s["step"] == "add_quantum"
                     else chain.remove_quantum())
        return chain

    def _then(self, step: Step) -> "Chain":
        return Chain(self.steps + (step,))

    def add_quantum(self) -> "Chain":
        """One quantum added, detected -- a photon absorbed, a Stokes photon
        scattered, an electron losing ``hbar omega`` (science § 2e.2):
        ``p'_n = n p_(n-1) / (<n> + 1)``; refused on a classical start, as
        every chain's steps are checked (:func:`_checked_steps`)."""
        return self._then(_step("add_quantum"))

    def remove_quantum(self) -> "Chain":
        """One quantum removed, detected -- an anti-Stokes photon scattered,
        an electron gaining ``hbar omega``: ``p'_n = (n + 1) p_(n+1) /
        <n>``.  Refused by name on a classical start (:func:`_checked_steps`)
        and on a state that holds none."""
        if not self.is_classical and _typed_as_holding_none(self.reduced()):
            raise ValueError(_HOLDS_NONE)
        return self._then(_step("remove_quantum"))

    def reduced(self) -> "Chain":
        """THE CHAIN IN ITS ONE FORM (`engines/vibration-frames.md` § 9.1):
        the ground state, or a level, with quanta added or removed straight
        after it, that level -- a quantum removed from the ground state
        refused by name.  A thermal state
        keeps its temperature, 0 K too: a thermal state is stated by it
        (its distribution the ground state's)."""
        out: list = []
        for s in self.steps:
            if s.name in ("add_quantum", "remove_quantum") and out and (
                    out[-1].name in ("ground", "level")):
                n = 0 if out[-1].name == "ground" else out[-1].get("quanta")
                n += 1 if s.name == "add_quantum" else -1
                if n < 0:
                    raise ValueError(_HOLDS_NONE)
                out[-1] = _step("ground") if n == 0 else _step("level",
                                                              quanta=n)
                continue
            if s.name == "level" and s.get("quanta") == 0:
                s = _step("ground")
            out.append(s)
        return Chain(tuple(out))

    def on(self, oscillator: Oscillator,
           temperature_k: Optional[float] = None,
           precision: float = _PRECISION) -> "OscillatorState":
        """THE STATE THIS CHAIN IS on ``oscillator``: a thermal start's mean
        quanta from the oscillator's frequency; a temperature left open --
        a thermal or classical start that names neither a temperature nor a
        mean -- taken from ``temperature_k``, refused by name where none is
        given.  A start that names its own temperature is given no other: a
        ``temperature_k`` that differs is refused by name; one that names
        its mean is refused any ``temperature_k``, the two naming one start
        twice; a chain with no thermal start (the ground state, a level)
        has no use for one.
        ``precision``: the share of the averages the levels a list leaves
        out may hold (:data:`_PRECISION`; science § 2e.4, § 2e.6) -- a
        number above 0 and below 1."""
        if not isinstance(oscillator, Oscillator):
            raise ValueError(f"a chain is placed on an Oscillator; got "
                             f"{oscillator!r}")
        share = _finite_non_negative(precision)
        if share is None or not 0.0 < share < 1.0:
            raise ValueError(f"precision is a share of the averages, above 0 "
                             f"and below 1; got {precision!r}")
        start = self.start
        if temperature_k is not None and not self.temperature_open and (
                start.name in ("thermal", "classical_thermal")):
            given = checked_temperature_k(temperature_k)
            own = start.get("temperature_k")
            if own is None:
                raise ValueError(
                    f"the chain's thermal start names its own mean quanta, "
                    f"{start.get('mean_quanta'):g}; a temperature given "
                    f"here, {given:g} K, would name another -- give one")
            if own != given:
                raise ValueError(
                    f"the chain's thermal start names its own temperature, "
                    f"{own:g} K; {given:g} K given here conflicts with it -- "
                    f"give one")
        if self.temperature_open:
            if temperature_k is None:
                raise ValueError(
                    "the chain's thermal start names no temperature: give "
                    "one here, temperature_k")
            start = self.with_temperature(temperature_k).start
        # A PLACED THERMAL START HOLDS ITS MEAN, resolved from its
        # temperature where it names one -- a start named one way, as every
        # chain's is -- the temperature kept on the state beside it.
        placed_at = start.get("temperature_k")
        if start.name == "thermal":
            mean = (oscillator.bose_einstein_mean_quanta(placed_at)
                    if placed_at is not None else start.get("mean_quanta"))
            start = _step("thermal", mean_quanta=float(mean))
        state = OscillatorState(Chain((start,) + self.steps[1:]).reduced(),
                                oscillator, share, placed_at)
        # NOTHING TO REMOVE is judged on the start AS TYPED -- a thermal one
        # at its temperature, never at a mean computed from it, which rounds
        # to 0 on a mode too cold for a number (:func:`_typed_as_holding_none`).
        if (len(state.steps) > 1 and state.steps[1].name == "remove_quantum"
                and _typed_as_holding_none(Chain((state._typed_start,)))):
            raise ValueError(_HOLDS_NONE)
        return state


def _typed_as_holding_none(chain: Chain) -> bool:
    """Whether a reduced chain, AS TYPED, holds no quantum to remove: the
    ground state, or a thermal start alone at exactly 0 K, or of no mean
    quanta -- the person's numbers, never one computed (a thermal state at
    4 K whose mean is below what a number holds is above 0 K, and a quantum
    removed from it is its limit, the ground state's list:
    :func:`_heat_list`)."""
    if chain.steps == (_step("ground"),):
        return True
    if len(chain.steps) != 1 or chain.start.name != "thermal":
        return False
    t = chain.start.get("temperature_k")
    return t == 0.0 if t is not None else chain.start.get("mean_quanta") == 0.0


def ground() -> Chain:
    """The ground state, ``|0>``: the zero-point motion alone."""
    return Chain((_start_step("ground", {}),))


def level(n: int) -> Chain:
    """The oscillator's level ``n``, ``|n>``: ``n`` quanta exactly -- a
    whole number, 0 or more."""
    return Chain((_start_step("level", {"quanta": n}),))


def thermal(temperature_k: Optional[float] = None,
            mean_quanta: Optional[float] = None) -> Chain:
    """The thermal state -- at ``temperature_k``, or of ``mean_quanta``
    quanta on average (many quanta added at random and relaxed, science
    § 2a.1), or neither: the temperature left open, taken where the chain
    is placed.  Both at once are refused by name."""
    return Chain((_start_step("thermal", {"temperature_k": temperature_k,
                                          "mean_quanta": mean_quanta}),))


def classical_thermal(temperature_k: Optional[float] = None) -> Chain:
    """The classical bell curve at ``temperature_k`` -- no zero-point motion,
    at 0 K a point at rest (science § 2b.3) -- or its temperature left
    open."""
    return Chain((_start_step("classical_thermal",
                              {"temperature_k": temperature_k}),))


# --------------------------------------------------------------------- #
#  A state: a chain on an oscillator                                     #
# --------------------------------------------------------------------- #

def _log_weights(n: np.ndarray, events: Tuple[str, ...]) -> np.ndarray:
    """``ln c(n)`` for a thermal start with ``events`` after it, in order
    (science § 2e.2): the list is ``p_n ∝ c(n) t^n``, ``c = 1`` on ``n >=
    0`` for the thermal state, and each quantum added ``c'(n) = n c(n -
    1)``, each removed ``c'(n) = (n + 1) c(n + 1)`` -- the powers of ``t``
    each event shifts in or out are one factor, the mean the step divides by
    cancelled.  ``-inf`` where ``c`` is 0."""
    at = n.astype(float)
    logs = np.zeros(at.shape)
    alive = np.ones(at.shape, dtype=bool)
    for event in reversed(events):
        factor = at if event == "add_quantum" else at + 1.0
        at = at - 1.0 if event == "add_quantum" else at + 1.0
        alive &= factor > 0.0
        logs += np.log(np.where(factor > 0.0, factor, 1.0))
    return np.where(alive & (at >= 0.0), logs, -np.inf)


def _heat_list(mean: float, events: Tuple[str, ...], order: int,
               precision: float) -> np.ndarray:
    """A thermal state of ``mean`` quanta and the quanta added and removed
    after it, by one formula (science § 2e.2), ``t = mean
    / (mean + 1)``: ``p_n ∝ c(n) t^n`` (:func:`_log_weights`) -- ``k``
    removals alone ``C(n + k, k) (1 - t)^(k+1) t^n``.  Nothing is divided by
    the mean, so a thermal state whose mean is below what a number holds
    (3000 cm-1 at 4 K, ``t`` rounded to 0) gives the state's limit, its
    lowest level ``c`` holds.  Built out until the levels beyond hold less
    than :data:`_CUT_PLACED_WITHIN` of ``precision`` of ``<x^order>``,
    uncut (science § 2e.4: the cut is taken on the final list): past the
    largest term each term ``T_n = p_n (2n + 1)^(order/2)`` is the last
    times ``r_n = t c(n + 1)/c(n) ((2n + 3)/(2n + 1))^(order/2)``, which
    falls toward ``t``, so the levels beyond ``n`` hold at most ``T_n r_n /
    (1 - r_n)``."""
    t = mean / (mean + 1.0)
    size = 64 + 2 * len(events)
    while True:
        n = np.arange(size + 1)
        logc = _log_weights(n, events)
        first = int(np.flatnonzero(np.isfinite(logc))[0])
        if t == 0.0:
            p = np.zeros(first + 1)
            p[first] = 1.0
            return p
        logp = logc + (n - first) * math.log(t)
        p = np.exp(logp - logp[np.isfinite(logp)].max())
        terms = p * (2.0 * n + 1.0) ** (order / 2.0)
        top = int(np.argmax(terms))
        ratio = (t * np.exp(logc[-1] - logc[-2])
                 * ((2.0 * size + 1.0) / (2.0 * size - 1.0)) ** (order / 2.0))
        if (top < size - 1 and ratio < 1.0
                and terms[-1] * ratio / (1.0 - ratio)
                < _CUT_PLACED_WITHIN * precision * terms.sum()):
            return p / p.sum()
        size *= 2


def _levels_kept(p: np.ndarray, order: int, precision: float) -> int:
    """THE CUT (science § 2e.4): how many levels a final list keeps -- the
    levels beyond hold less than ``precision`` of ``<x^order>``."""
    terms = p * (2.0 * np.arange(p.size) + 1.0) ** (order / 2.0)
    tail = np.cumsum(terms[::-1])[::-1]      # tail[N]: the levels N and up
    small = np.flatnonzero(tail < precision * terms.sum())
    return max(1, int(small[0]) if small.size else p.size)


@functools.lru_cache(maxsize=64)
def _evolve(chain: Chain, order: int, precision: float,
            typed_start: Step) -> np.ndarray:
    """The list a placed, reduced quantum ``chain`` makes (science § 2e.2):
    a level, ``p_n = 1`` at it; a thermal start and the quanta added and
    removed after it, from one formula (:func:`_heat_list`) -- a thermal
    start typed at exactly 0 K the ground state, its events taken one by
    one (a quantum added ``p'_n = n p_(n-1)``, one removed ``p'_n = (n + 1)
    p_(n+1)``, each renormalised), a removal it cannot make refused --
    judged on ``typed_start``, the start as the person typed it
    (:attr:`OscillatorState._typed_start`).  Cut where its levels hold less
    than ``precision`` of ``<x^order>`` (:func:`_levels_kept`),
    read-only."""
    start, events = chain.start, tuple(s.name for s in chain.steps[1:])
    if start.name == "level":
        p = np.zeros(start.get("quanta") + 1)
        p[-1] = 1.0
    elif start.name == "ground" or _typed_as_holding_none(
            Chain((typed_start,))):
        p = np.array([1.0])                 # exactly 0 K, as typed
        for event in events:
            if event == "add_quantum":
                q = np.zeros(p.size + 1)
                q[1:] = np.arange(1, p.size + 1) * p
            else:
                q = np.arange(1, p.size) * p[1:]
            if not q.sum() > 0.0:
                raise ValueError(_HOLDS_NONE)
            p = q / q.sum()
    else:
        p = _heat_list(start.get("mean_quanta"), events, order, precision)
    p = p[:_levels_kept(p, order, precision)]
    p = p / p.sum()
    p.flags.writeable = False
    return p


def _family(chain: Chain) -> str:
    """THE CLOSED FAMILY a placed, reduced chain is of, found once for the
    routes (`distribution.of_state`, `engines/vibration-frames.md` § 9.1's
    diagram) and the means (:func:`_closed_mean`): ``"bell"`` a start
    alone, a level aside; ``"quantum added"`` and ``"quantum removed"`` a
    thermal start and one event; ``"level"`` a level alone -- or ``""``, any
    other chain: its list."""
    names = tuple(s.name for s in chain.steps)
    first, tail = names[0], names[1:]
    if first == "level":
        return "level" if not tail else ""
    if not tail:
        return "bell"
    if first == "thermal" and tail == ("add_quantum",):
        return "quantum added"
    if first == "thermal" and tail == ("remove_quantum",):
        return "quantum removed"
    return ""


def _closed_mean(chain: Chain) -> Optional[float]:
    """A placed, reduced quantum chain's mean quanta where it is known in
    closed form (science § 2e.1, § 2e.2): a level ``n``; the ground state 0
    and a thermal state ``nbar``; a thermal state with one quantum added ``2
    nbar + 1``, one removed ``2 nbar`` (a thermal start at exactly 0 K
    refused before: :meth:`Chain.on`) -- ``None`` for any other."""
    family = _family(chain)
    start = chain.start
    nbar = start.get("mean_quanta", 0.0)
    if family == "level":
        return float(start.get("quanta"))
    if family == "bell":
        return nbar
    if family == "quantum added":
        return 2.0 * nbar + 1.0
    if family == "quantum removed":
        return 2.0 * nbar
    return None


@dataclass(frozen=True)
class OscillatorState:
    """A CHAIN PLACED ON AN OSCILLATOR (:meth:`Chain.on`) -- its reduced
    ``chain``, a thermal start's mean resolved (the start holds its mean
    alone, named one way as every chain's start is), its ``oscillator``,
    the ``precision`` its list is cut to, and ``temperature_k``, the
    temperature of its start -- a thermal one's it was placed at, a
    classical one's -- or ``None``."""
    chain: Chain
    oscillator: Oscillator
    precision: float = _PRECISION
    temperature_k: Optional[float] = None

    @property
    def steps(self) -> Tuple[Step, ...]:
        return self.chain.steps

    @property
    def is_classical(self) -> bool:
        return self.chain.is_classical

    def populations_for(self, order: int) -> np.ndarray:
        """Its list, cut for the averages up to ``<x^order>`` (science
        § 2e.4) -- a quantum state's; a classical one counts no quanta."""
        if self.is_classical:
            raise ValueError(_COUNTS_NO_QUANTA)
        return _evolve(self.chain, _whole(order, "an average's order", 0),
                       self.precision, self._typed_start)

    @property
    def _typed_start(self) -> Step:
        """Its start as the person typed it, temperature filled where it was
        left open: a thermal start placed at a temperature that temperature
        (its chain holding the mean resolved from it), else the chain's own
        start -- what a removal from a state that holds none is judged on,
        never a mean computed (:func:`_typed_as_holding_none`)."""
        if self.chain.start.name == "thermal" and self.temperature_k is not None:
            return _step("thermal", temperature_k=self.temperature_k)
        return self.chain.start

    @property
    def populations(self) -> np.ndarray:
        """Its list -- the chance of each number of quanta -- cut for the
        averages the offered frames match (:data:`_DEFAULT_ORDER`)."""
        return self.populations_for(_DEFAULT_ORDER)

    @property
    def _family(self) -> str:
        """Its closed family (:func:`_family`)."""
        return _family(self.chain)

    @property
    def mean_quanta(self) -> Optional[float]:
        """Its mean number of quanta -- in closed form where it is known
        (:func:`_closed_mean`), else from its list; ``None`` for a classical
        state."""
        if self.is_classical:
            return None
        closed = _closed_mean(self.chain)
        if closed is not None:
            return float(closed)
        p = self.populations
        return float((np.arange(p.size) * p).sum())

    @property
    def words(self) -> str:
        """The state in words, each step saying itself, its thermal start's
        mean resolved (*thermal at 300 K (0.621 mean quanta); one quantum
        added*) -- :func:`_steps_words`."""
        return _steps_words(self.chain.steps, None, self.temperature_k)

    def words_with_source(self, source: Optional[str]) -> str:
        """Its words, its start's temperature said as ``source``'s
        (*thermal at 300 K (0.621 mean quanta), the record's temperature;
        one quantum added*); ``None`` says none."""
        return _steps_words(self.chain.steps, source, self.temperature_k)


def _steps_words(steps: Tuple[Step, ...], source: Optional[str] = None,
                 placed_at: Optional[float] = None) -> str:
    """A CHAIN'S STEPS IN WORDS, the one wording of a state's steps -- a
    chain's (:attr:`Chain.words`) and a placed state's
    (:attr:`OscillatorState.words`), each step saying what it states: a
    thermal start its temperature and, placed, its mean quanta; a start
    left open says so; ``source``, where an open start's temperature was
    taken from, said right after that temperature (*…, the record's
    temperature*); ``placed_at``, the temperature a placed thermal start
    was placed at, its step holding the mean resolved from it."""
    taken = f", {source} temperature" if source else ""
    out = []
    for s in steps:
        if s.name == "ground":
            out.append("the ground state")
        elif s.name == "level":
            out.append(f"{_quanta_words(s.get('quanta'))} exactly")
        elif s.name == "thermal":
            t, n = s.get("temperature_k"), s.get("mean_quanta")
            t = placed_at if t is None else t
            mean = (None if n is None else
                    _quanta_words(n, kind="mean ", spec=".3g"))
            if t is not None:
                out.append(f"thermal at {t:g} K"
                           + (f" ({mean})" if mean is not None else "")
                           + taken)
            else:
                out.append(f"thermal, {mean}" if mean is not None
                           else "thermal, its temperature open")
        elif s.name == "classical_thermal":
            t = s.get("temperature_k")
            out.append("the classical bell curve "
                       + (f"at {t:g} K{taken}" if t is not None
                          else "at a temperature left open")
                       + ", no zero-point motion")
        elif s.name == "add_quantum":
            out.append("one quantum added")
        else:
            out.append("one quantum removed")
    return "; ".join(out)


#: WHAT THIS MODULE NEEDS BESIDE IT in a job's bundle -- the modules it
#: imports there by their bare names (`runwrap.bundle_members`).
TRAVELS_WITH = ("molbuilder.oscillator",)

__all__ = ["Step", "Chain", "OscillatorState", "ground", "level", "thermal",
           "classical_thermal"]
