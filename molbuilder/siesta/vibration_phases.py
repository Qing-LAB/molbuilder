"""A SIESTA VIBRATION AS ONE RUN OF POINTS -- the phases, each SIESTA call
a point of the run (`engines/vibration.md` § 5.11; the framework,
`execution/runs-of-points.md`).

MODULE  siesta.vibration_phases (floor 1 + stdlib; travels in
        ``mb_vibration.pyz`` and ``mb_monitor.pyz``,
        `runwrap.VIBRATION_COMPANIONS`, `runwrap.MONITOR_COMPANIONS`)
ROLE    the one table of the phases (:data:`PHASES`): what each phase's
        deck is, what turns it on, where its calls' coordinates and start
        come from, what a call must hold to be done, which warm-files
        section it reads, which phases it needs, what a relaunch does with
        a call not done, and which phase's step plans its calls -- read by
        prep (the decks and run scripts), the pair's declaration
        (`jobset.vibration_points`: launch's walk, `status`) and the run's
        own python (its steps).  The call (:class:`Call`), a point of the
        run; a call's entry in the run's index (:func:`point_entry`), the
        one shape launch and the force constants' step write; and the
        count in the vibration's words (:func:`count_words`, the kind
        table's ``count`` part).  The index itself, its stamps and its
        rules are every run of points' (`runpoints`).
USED-BY `jobset.prep`, `jobset.vibration_points`, `spectra.siesta_vibration`,
        `siesta.vibration_deck`, `mode_frames`, `spectra.displacement_sweep`,
        `monitor` (the count, through the index's part)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# TWO WAYS, because this module travels beside a job (`runwrap.
# VIBRATION_COMPANIONS`, `runwrap.MONITOR_COMPANIONS`), where the package
# is not installed.
try:                                       # inside molbuilder
    from ..calcdirs import RunPlace
    from ..paths import frame_index
    from ..runfiles import compose
    from ..runpoints import AGAIN, CONTINUE
except ImportError:                        # beside a job
    from calcdirs import RunPlace  # type: ignore[no-redef]
    from paths import frame_index  # type: ignore[no-redef]
    from runfiles import compose  # type: ignore[no-redef]
    from runpoints import AGAIN, CONTINUE  # type: ignore[no-redef]

#: The phases' names -- one spelling each, the folder below the stage and
#: below a run that holds the phase's deck, and a call's.
RELAX, FC, OPTICAL = "relax", "fc", "optical"


@dataclass(frozen=True)
class PhaseRule:
    """One phase of the run (`engines/vibration.md` § 5.11 § 1, § 3)."""
    name: str
    #: `spec_for`'s ``part`` -- which sections the phase's deck holds (§ 4).
    part: str
    #: The rung role the phase's deck renders -- the catalogue's `stages`
    #: vocabulary (the kind table's vibration roles, § 5.11 § 9): the one
    #: link between a phase's folder name and the items its deck reads.
    role: str
    #: The catalogue item that turns the phase off, and the value that does:
    #: ``None`` for a phase that always runs.
    switch: Optional[Tuple[str, Any]]
    #: Where a call's coordinates come from: ``structure`` (prep's file),
    #: the ``relax`` call's output, or the ``fc`` call's modes (frame 0 the
    #: reference geometry, then ±δQ along each picked mode).
    coordinates: str
    #: The call whose file seeds this phase's calls -- ``(phase, suffix)``,
    #: the density the SCF starts from -- or ``None``.
    seed: Optional[Tuple[str, str]]
    #: The files a call must hold to be done, by suffix on the label
    #: (`continuation.done`'s products); a relaxation holds none -- done is
    #: finished, converged or not (§ 5).
    products: Tuple[str, ...]
    #: Whether a call's VERDICT counts toward done -- its SCF converged
    #: (`_run_ending.not_done_why`): not a relaxation's, done finished,
    #: converged or not (§ 5, § 6).  Read by this machine's done door
    #: (`jobset.vibration_points`), and copied into each call's entry in
    #: the run's index, which the job judges it by (`runpoints.rows_of`).
    verdict_counts: bool
    #: The warm-files section a call of this phase reads, launched again
    #: not done (§ 6).
    warm_kind: str
    #: The phases a call of this one needs to have RAN (§ 5;
    #: `runs-of-points.md` § 1.2).
    needs: Tuple[str, ...]
    #: What a launch again does with a call not done (`runs-of-points.md`
    #: § 1.2): runs it ``again`` from its start, or ``continue``s it from
    #: its own last state, its job's warm files handed over (§ 6).
    relaunch: str = AGAIN
    #: The phase whose call's step PLANS this phase's calls in the job,
    #: each in a folder below a container named after the phase (§ 3) --
    #: ``None`` for a phase whose call prep prepares.
    planned_by: Optional[str] = None

    @property
    def planned(self) -> bool:
        """Whether the phase's calls are planned in the job."""
        return self.planned_by is not None


#: THE TABLE (§ 5.11 § 3, *Data, not code*): in the order the walk runs them.
PHASES: Tuple[PhaseRule, ...] = (
    PhaseRule(RELAX, part="relax", role="relaxation",
              switch=("already_relaxed", True),
              coordinates="structure", seed=None, products=(),
              verdict_counts=False, warm_kind="optimization", needs=(),
              relaunch=CONTINUE),
    PhaseRule(FC, part="fc", role="force_constants", switch=None,
              coordinates=RELAX, seed=(RELAX, ".DM"), products=(".FC",),
              verdict_counts=True, warm_kind="vibration", needs=(RELAX,)),
    PhaseRule(OPTICAL, part="optical", role="optical",
              switch=("compute_raman", False),
              coordinates=FC, seed=(FC, ".DM"), products=(".EPSIMG",),
              verdict_counts=True, warm_kind="optical", needs=(FC,),
              planned_by=FC),
)


#: THE INFRARED'S CHARGES (`engines/vibration.md` § 5.3, § 5.10 ②): each
#: partition the force-constant phase asks SIESTA for at every step when
#: Compute IR is on -- its keyword, at its value, written by the deck
#: (`vibration_deck.fc_block`) and read back by the finish -- beside the name
#: the output reader gives its blocks (`siesta_reader`).
IR_CHARGES: Tuple[Tuple[str, str, str], ...] = (
    ("Charge.Hirshfeld", "geometry", "hirshfeld"),
    ("Charge.Voronoi", "geometry", "voronoi"),
)


def phase(name: str) -> PhaseRule:
    """The phase named ``name``, refused by name when there is none."""
    for p in PHASES:
        if p.name == name:
            return p
    raise ValueError(f"no phase {name!r}: the phases are "
                     f"{', '.join(p.name for p in PHASES)}")


def phases_on(cfg) -> Tuple[PhaseRule, ...]:
    """The phases a stage runs, by each row's switch read off the stage's
    resolved config ``cfg`` (§ 5.11 § 1): a phase with no switch always;
    one whose switch holds its off value, not."""
    return tuple(p for p in PHASES
                 if p.switch is None
                 or getattr(cfg, p.switch[0], None) != p.switch[1])


@dataclass(frozen=True)
class Call:
    """ONE SIESTA CALL OF THE RUN -- a point of it (`jobset.points`): its
    phase, and for an optical call which frame it is -- ``name`` its
    folder's (``f000``, ``f001``, …), the mode (1-based; ``None`` for frame
    0, the equilibrium), the side (+1 / −1 / 0), the mode's frequency and
    the displacement ``δQ`` (amu^½ Å).  A relaxation's and the force
    constants' ``name`` is the phase's own.  ``name`` ``None`` is the
    phase's folder as prep writes it -- its deck and run script, what every
    call of the phase copies (§ 2)."""
    phase: str
    name: Optional[str] = None
    mode: Optional[int] = None
    side: Optional[int] = None
    frequency_cm1: Optional[float] = None
    dq_amu12_ang: Optional[float] = None
    #: How many calls the phase holds, said in the call's words, never part
    #: of which call it is.
    of: Optional[int] = field(default=None, compare=False)

    @property
    def rel(self) -> str:
        """Its folder below the stage or a run: the phase's own name for the
        relaxation, the force constants and a phase folder; ``optical/f001``
        for an optical call."""
        if self.name is None or self.name == self.phase:
            return self.phase
        return f"{self.phase}/{self.name}"

    @property
    def prepared(self) -> str:
        """Its prepared folder below the stage -- the phase's, which holds
        the deck and run script every call of the phase runs
        (`runs-of-points.md` § 1.1)."""
        return self.phase

    def words(self) -> str:
        """The call as a message names it: *the relaxation*, *the force
        constants*, *the equilibrium*, *mode 7 (4271 cm⁻¹), +δQ*."""
        if self.phase == RELAX:
            return "the relaxation"
        if self.phase == FC:
            return "the force constants"
        if self.mode is None:
            return "the equilibrium"
        side = "+δQ" if (self.side or 0) > 0 else "−δQ"
        freq = (f" ({self.frequency_cm1:.0f} cm⁻¹)"
                if self.frequency_cm1 is not None else "")
        return f"mode {self.mode}{freq}, {side}"

    def to_wire(self) -> Dict[str, Any]:
        """The call on the wire -- one form for every answer that names a
        point: ``{phase, call, mode, side, frequency_cm1, dq_amu12_ang}``;
        ``frame``, ``of`` and ``bias_v`` beside them, as every point's wire
        form carries them (`transport.stages.Point.to_wire`) -- a frame
        index for an optical call, no voltage."""
        frame = (frame_index(self.name) if self.phase == OPTICAL and self.name
                 else None)
        return {"phase": self.phase, "call": self.name, "mode": self.mode,
                "side": self.side, "frequency_cm1": self.frequency_cm1,
                "dq_amu12_ang": self.dq_amu12_ang,
                "frame": frame, "of": self.of, "bias_v": None}

    def facts(self) -> Dict[str, Any]:
        """What the call is, as its entry in the run's index keeps it
        (`runs-of-points.md` § 2.3, ``facts``)."""
        return {"name": self.name, "mode": self.mode, "side": self.side,
                "frequency_cm1": self.frequency_cm1,
                "dq_amu12_ang": self.dq_amu12_ang}

    @classmethod
    def from_entry(cls, entry: Dict[str, Any], *, of: Optional[int] = None
                   ) -> "Call":
        """The call an entry of the run's index is."""
        f = entry.get("facts") or {}
        return cls(phase=str(entry["phase"]), name=f.get("name"),
                   mode=f.get("mode"), side=f.get("side"),
                   frequency_cm1=f.get("frequency_cm1"),
                   dq_amu12_ang=f.get("dq_amu12_ang"), of=of)


# --------------------------------------------------------------------- #
#  A call in the run's index (`runpoints`)                               #
# --------------------------------------------------------------------- #

def calls_of(index: Dict[str, Any]) -> List[Call]:
    """Every call the run's index lists, in its order, each with how many
    its phase holds."""
    rows = index.get("points") or []
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r["phase"]] = counts.get(r["phase"], 0) + 1
    return [Call.from_entry(r, of=counts[r["phase"]]) for r in rows]


def must_hold(call: Call, label: str) -> List[Tuple[str, str]]:
    """What a call must hold to be done, each with the words its absence is
    said in: its phase's products, named on the label
    (`runfiles.compose`); none for a relaxation, done finished (§ 5)."""
    return [(compose(label, suf), compose(label, suf))
            for suf in phase(call.phase).products]


def point_entry(call: Call, label: str) -> Dict[str, Any]:
    """THE CALL'S ENTRY in the run's index, as launch and the force
    constants' step write it (`runpoints.add_point`; `runs-of-points.md`
    § 2.3): its folder and words, its phase -- its round too -- what it
    must hold and whether its verdict counts, what it is.  Its needs, its
    planner and where it was taken over from are the writer's."""
    return {"rel": call.rel, "words": call.words(), "phase": call.phase,
            "round": call.phase, "must_hold": must_hold(call, label),
            "verdict_counts": phase(call.phase).verdict_counts,
            "facts": call.facts()}


def count_words(rows, facts) -> str:
    """THE RUN'S COUNT in the vibration's words (the kind table's
    ``count`` part, `runs-of-points.md` § 4.7): *relax done · fc done ·
    optical 7 of 11 done (4 taken over, 2 failed)* -- the calls taken over,
    and those that ended without being done, each said where there are any;
    a planned phase with no call yet, where the run walks it (``facts``'
    ``phases``), *after fc* -- or, its planner done, *no mode picked*.
    ``rows`` are the job's (`runpoints.rows_of`) or `status`'s
    (`continuation.point_rows`): one shape."""
    phases = tuple((facts or {}).get("phases") or ())
    try:                                   # inside molbuilder
        from ..runpoints import NOT_RUN
    except ImportError:                    # beside a job, the monitor's
        from runpoints import NOT_RUN  # type: ignore[no-redef]
    parts: List[str] = []
    by: Dict[str, list] = {}
    for r in rows:
        by.setdefault(r["phase"], []).append(r)
    for p in PHASES:
        got = by.get(p.name)
        if not got:
            if p.planned and by and p.name in phases:
                # BEFORE THE CALL THAT PLANS IT, or after it with no mode
                # picked -- the selection's answer (§ 5.11 § 7).
                planner_done = any(r["done"] for r in by.get(p.planned_by)
                                   or [])
                parts.append(f"{p.name}: no mode picked" if planner_done
                             else f"{p.name}: after {p.planned_by}")
            continue
        n_done = sum(bool(r["done"]) for r in got)
        # TAKEN OVER is the index's ``taken_from`` (`continuation.taken_over`),
        # the key the job's rows and the status door's rows share.
        taken = sum(bool(r.get("taken_from")) for r in got if r["done"])
        # FAILED -- it ran and ended with an error -- and NOT RUN -- its
        # ended walk never started it -- each its row's ``state``, one rule
        # for the job's rows and `status`'s, as a transport run's count
        # says them (`transport.run_words.count_words`).
        failed = sum(1 for r in got if r.get("state") == "failed")
        not_run = sum(1 for r in got if r.get("state") == NOT_RUN)
        # RAN TO ITS END AND NOT DONE (`runs-of-points.md` § 1.3).
        not_done = sum(1 for r in got
                       if r.get("state") == "finished" and not r["done"])
        if len(got) == 1:
            # ONE CALL: done, not done, or its row's own state word.
            parts.append(f"{p.name} "
                         f"{'done' if n_done else 'not done' if not_done else got[0]['state']}")
        else:
            notes = ([f"{taken} taken over"] if taken else []) + (
                [f"{failed} failed"] if failed else []) + (
                [f"{not_run} not run"] if not_run else []) + (
                [f"{not_done} not done"] if not_done else [])
            parts.append(f"{p.name} {n_done} of {len(got)} done"
                         + (f" ({', '.join(notes)})" if notes else ""))
    return " · ".join(parts) if parts else "no call yet"


#: AN .EPSIMG ROW'S BYTES PER COLUMN, as SIESTA writes it list-directed:
#: the photon energy, then eps2 per spin channel, 26 characters each and the
#: line's end -- 53 bytes a row with one channel, measured on the CO2
#: end-to-end run (2,650,217 bytes over 50,009 lines, 2026-10-26).
EPSIMG_COLUMN_BYTES = 26


def run_size_words(cfg, n_free: int, n_modes: int) -> str:
    """THE COUNT AND THE SIZES OF THE RUN the stage's settings make, in
    words -- what prep says beside the wall (`engines/vibration.md` § 5.11
    § 9): the relaxation up to ``relax_steps`` geometry steps a try (its
    retries the run card's); the force constants ``1 + 6·n_free`` SCFs;
    Raman ``1 + 2·N`` calls, ``N`` every one of the ``n_modes`` the
    run reports under *Raman for* ``all`` (prep's count, the rank rule,
    `validation.spectra.modes_of`), at most that within a range, which only
    the run can count; each ``.EPSIMG``'s size from its photon energies
    (``Optical.Energy.Maximum`` over a step of ``Optical.Broaden / 10``,
    `Src/optical.F`).  Only the phases the settings turn on
    (:func:`phases_on`).  The wall stays the person's: nothing here judges
    it."""
    def _n(k: int, word: str) -> str:
        return f"{k} {word}{'s' if k != 1 else ''}"
    on = {p.name for p in phases_on(cfg)}
    parts = []
    if RELAX in on:
        parts.append(f"relax up to {int(cfg.relax_steps)} geometry steps a "
                     f"try")
    parts.append(f"fc {_n(1 + 6 * n_free, 'SCF')} (1 + 6 × "
                 f"{_n(n_free, 'free atom')})")
    if OPTICAL in on:
        if str(cfg.raman_modes) == "all":
            calls = (f"{_n(1 + 2 * n_modes, 'call')} (1 + 2 × "
                     f"{_n(n_modes, 'mode')})")
        else:
            calls = (f"up to {_n(1 + 2 * n_modes, 'call')} (1 + 2 × "
                     f"the modes within the range, at most {n_modes})")
        broaden = float(cfg.optical_broaden)
        if broaden > 0:
            rows = int(round(float(cfg.optical_energy_max)
                             / (broaden / 10.0))) + 1
            one = rows * (2 * EPSIMG_COLUMN_BYTES + 1) / 1e6
            more = rows * EPSIMG_COLUMN_BYTES / 1e6
            calls += (f", each .EPSIMG about {one:.1f} MB ({rows:,} "
                      f"photon energies, one spin channel; {more:.1f} "
                      f"MB more with two)")
        parts.append(f"optical {calls}")
    return "the run: " + " · ".join(parts)


def force_constants_call(run_dir) -> Path:
    """THE FORCE-CONSTANT CALL of the run at ``run_dir`` -- its folder below
    the run (`Call.rel`): the call whose deck measured the force constants
    and which leaves the engine's `.FC` (§ 5.11 § 2), the one a reader of
    the run's force constants or of the deck they were measured with
    takes (the sweep summary, a mode's frame set)."""
    return RunPlace.of(run_dir).point(Call(FC, FC).rel)


__all__ = ["RELAX", "FC", "OPTICAL", "PhaseRule", "PHASES", "phase",
           "phases_on", "IR_CHARGES", "Call", "calls_of", "must_hold",
           "point_entry", "count_words", "run_size_words",
           "force_constants_call"]
