"""A SIESTA VIBRATION AS ONE RUN IN PHASES -- the phases, each SIESTA call
a point of the run, and the run's index (`engines/vibration.md` § 5.11).

MODULE  siesta.vibration_phases (floor 1 + stdlib; travels in
        ``mb_vibration.pyz`` and ``mb_monitor.pyz``,
        `runwrap.VIBRATION_COMPANIONS`, `runwrap.MONITOR_COMPANIONS`)
ROLE    the one table of the phases (:data:`PHASES`): what each phase's
        deck is, what turns it on, where its calls' coordinates and seed
        come from, what a call must hold to be done, which warm-files
        section it reads, which phases it needs -- read by prep (the decks
        and run scripts), launch (the walk, the calls it opens and takes
        over), the run's own python (the plan, record and finish verbs)
        and every reader of the run's state.  The index -- one per run, at
        its root, named for its first try
        (``<label>_<stage>-run0.phases.json``, :func:`index_path`,
        :func:`write_index` / :func:`read_index`) -- lists every call: what it is, from what, how it
        ended -- written by launch for the calls it opens and by the run's
        own python for the rest, rewritten whole; its rows by the one
        done rule (:func:`rows_of`) and their count (:func:`summary`) are
        what the run's python, its finish and a member monitor's line say
        (`run-reports.md` § 2.8).  The kind's declaration
        as a rung of points is floor 4's, `jobset.vibration_points`, which
        asks this module's calls, their state and the run's count.
USED-BY `jobset.prep`, `jobset.submit`, `jobset.vibration_points`,
        `spectra.siesta_vibration`, `parse.sidecars.vibration_phases`,
        `monitor`
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# TWO WAYS, because this module travels beside a job (`runwrap.
# VIBRATION_COMPANIONS`, `runwrap.MONITOR_COMPANIONS`), where the package
# is not installed.
try:                                       # inside molbuilder
    from .. import runrecord as _rr
    from ..runfiles import ROLE_PHASES, compose
except ImportError:                        # beside a job
    import runrecord as _rr  # type: ignore[no-redef]
    from runfiles import ROLE_PHASES, compose

SCHEMA = "molbuilder/vibration-phases@1"

#: What one point of the run is called -- each SIESTA call (`run-reports.md`
#: § 2.8, `call_unit`): read by the run of points' declaration and the
#: monitor.
UNIT = "call"

#: The run's index written whole, and a call not done in the words a failed
#: part and a failed finish name it by -- every run of points' own
#: (`runrecord`).
write_index = _rr.write_index
not_done_words = _rr.not_done_words

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
    #: (`jobset.vibration_points`) and by the job's own record step.
    verdict_counts: bool
    #: The warm-files section a call of this phase reads, launched again
    #: not done (§ 6).
    warm_kind: str
    #: The phases a call of this one needs ended normally (§ 5).
    needs: Tuple[str, ...]
    #: Whether the phase's calls are PLANNED IN THE JOB, each in a folder
    #: below a container named after the phase (§ 3): a member monitor's
    #: way up to the run is a call's, not the phase folder's (§ 7a).
    planned: bool = False


#: THE TABLE (§ 5.11 § 3, *Data, not code*): in the order the walk runs them.
PHASES: Tuple[PhaseRule, ...] = (
    PhaseRule(RELAX, part="relax", role="relaxation",
              switch=("already_relaxed", True),
              coordinates="structure", seed=None, products=(),
              verdict_counts=False, warm_kind="optimization", needs=()),
    PhaseRule(FC, part="fc", role="force_constants", switch=None,
              coordinates=RELAX, seed=(RELAX, ".DM"), products=(".FC",),
              verdict_counts=True, warm_kind="vibration", needs=(RELAX,)),
    PhaseRule(OPTICAL, part="optical", role="optical",
              switch=("compute_raman", False),
              coordinates=FC, seed=(FC, ".DM"), products=(".EPSIMG",),
              verdict_counts=True, warm_kind="optical", needs=(FC,),
              planned=True),
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
    def to_run(self) -> str:
        """The way from the call's folder up to its run's: ``..`` per
        level."""
        return "/".join([".."] * len(self.rel.split("/")))

    def words(self) -> str:
        """The call as a message names it: *the relaxation*, *the force
        constants*, *frame 0, the equilibrium*, *mode 7 (4271 cm⁻¹), +δQ*."""
        if self.phase == RELAX:
            return "the relaxation"
        if self.phase == FC:
            return "the force constants"
        if self.mode is None:
            return "frame 0, the equilibrium"
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
        frame = (int(self.name[1:]) if self.phase == OPTICAL and self.name
                 and self.name.startswith("f") and self.name[1:].isdigit()
                 else None)
        return {"phase": self.phase, "call": self.name, "mode": self.mode,
                "side": self.side, "frequency_cm1": self.frequency_cm1,
                "dq_amu12_ang": self.dq_amu12_ang,
                "frame": frame, "of": self.of, "bias_v": None}

    def to_dict(self) -> Dict[str, Any]:
        return {"phase": self.phase, "name": self.name, "mode": self.mode,
                "side": self.side, "frequency_cm1": self.frequency_cm1,
                "dq_amu12_ang": self.dq_amu12_ang}

    @classmethod
    def from_dict(cls, d: Dict[str, Any], *, of: Optional[int] = None
                  ) -> "Call":
        return cls(phase=str(d["phase"]), name=d.get("name"),
                   mode=d.get("mode"), side=d.get("side"),
                   frequency_cm1=d.get("frequency_cm1"),
                   dq_amu12_ang=d.get("dq_amu12_ang"), of=of)


# --------------------------------------------------------------------- #
#  The index -- every call of the run, what it is, how it ended           #
# --------------------------------------------------------------------- #

def index_path(run_dir, names) -> Path:
    """Where the run's index lies: at the run's root, named for its first
    try, ``<label>_<stage>-run0.phases.json`` (`runfiles.ROLE_PHASES`,
    `runrecord.index_path`)."""
    return _rr.index_path(run_dir, names, ROLE_PHASES)


def empty_index(phases) -> Dict[str, Any]:
    """An index with no call yet -- what launch starts from: ``phases`` the
    ones the run walks (its job's row, `model.Job.phases`), which the
    count's words read (:func:`summary`).  The run's notifications are the
    monitor's own record, not the index's (`runrecord.notified`, § 5.11
    § 7)."""
    return {"schema": SCHEMA, "phases": list(phases), "calls": [],
            "finish": None}


def read_index(path) -> Dict[str, Any]:
    """The index at ``path``, refused by name when it is not one of this
    schema (`runrecord.read_index`)."""
    return _rr.read_index(path, SCHEMA)


def calls_of(index: Dict[str, Any]) -> List[Call]:
    """Every call the index lists, in its order, each with how many its
    phase holds."""
    rows = index.get("calls") or []
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r["phase"]] = counts.get(r["phase"], 0) + 1
    return [Call.from_dict(r, of=counts[r["phase"]]) for r in rows]


def entry_of(index: Dict[str, Any], call: Call) -> Optional[Dict[str, Any]]:
    """The index's entry for ``call``, or ``None``."""
    for r in index.get("calls") or []:
        if r.get("phase") == call.phase and r.get("name") == call.name:
            return r
    return None


def add_call(index: Dict[str, Any], call: Call, **facts) -> Dict[str, Any]:
    """``call`` added to the index with ``facts`` (its deck's and
    coordinates' sha256, its seed, where it was computed) -- or its entry
    updated, where it is listed already."""
    row = entry_of(index, call)
    if row is None:
        row = call.to_dict()
        index.setdefault("calls", []).append(row)
    row.update(facts)
    return row


# --------------------------------------------------------------------- #
#  The run's count, size and files, as its declaration as a rung of     #
#  points asks them (`jobset.vibration_points`)                          #
# --------------------------------------------------------------------- #

def rows_of(index: Dict[str, Any], run_dir, label: str) -> List[Dict[str, Any]]:
    """EACH CALL THE INDEX LISTS, BY THE ONE DONE RULE (§ 5.11 § 6, § 7;
    `continuation.done` asks the same pieces): done when it ended with exit
    0, its record step did not fail, its SCF converged where its phase's
    verdict counts (`PhaseRule.verdict_counts`, the verdict its record step
    read) and it holds its phase's products (:func:`products_of`) -- the
    last two said in the rule's own words (`_run_ending.not_done_why`).
    Each row carries the call's words, its start and end as the index
    stamped them, and the run it was taken over from, where it was --
    what :func:`summary` counts, the run's own record step asks (the walk
    and the run's state) and a member monitor reads (`run-reports.md`
    § 2.8).  ``run_dir`` is the run's folder, ``label`` the calculation's."""
    try:                                   # inside molbuilder
        from ..parse.engines._run_ending import not_done_why
    except ImportError:                    # beside a job
        from _run_ending import not_done_why  # type: ignore[no-redef]
    rows = []
    for row in index.get("calls") or []:
        call = Call.from_dict(row)
        # A CALL THE ENDED WALK NEVER RAN (:func:`close`) ended with it,
        # not done -- its why the index's.
        never = row.get("not_run")
        ran = row.get("ended") is not None
        ended = ran or never is not None
        said = row.get("verdict")
        why = never or "; ".join(w for w in (
            (f"exit {row.get('exit')}" if ran
             and row.get("exit") not in (0, "0") else None),
            row.get("record_error")) if w) or None
        if ended and why is None:
            why = not_done_why(
                (said["of"], bool(said["converged"])) if said else None,
                counts=phase(call.phase).verdict_counts,
                missing=[name for name in products_of("", label, None,
                                                      point=call)
                         if not (Path(run_dir) / call.rel / name).is_file()])
        ok = ended and why is None
        rows.append({"phase": row["phase"], "name": row.get("name"),
                     "rel": call.rel, "words": call.words(), "done": ok,
                     "ended": ended, "started": row.get("started"),
                     "ended_at": row.get("ended"),
                     "started_from": row.get("taken_from"),
                     "taken_from": row.get("taken_from"),
                     "state": ("finished" if ok else
                               "failed" if ended else "pending"),
                     "why": why, "not_run": never})
    return rows


def still_to_run(index: Dict[str, Any], rows) -> List[Dict[str, Any]]:
    """THE CALLS THE WALK MAY STILL RUN (§ 5.11 § 5): not ended, and every
    phase each needs -- of those the run walks -- holding a call done
    (`PhaseRule.needs`, the walk's).  A call whose need did not end done is
    not run, so the finish runs once none is left."""
    walked = set(index.get("phases") or ())
    done = {r["phase"] for r in rows if r["done"]}
    return [r for r in rows if not r["ended"]
            and all(n in done for n in phase(r["phase"]).needs
                    if n in walked)]


def run_state(rows) -> "tuple[str, Optional[str]]":
    """THE RUN'S STATE, once no call is left to run, by the one done rule:
    finished when every call is done (§ 5.11 § 7), else failed naming the
    first call not done and how many when more than one is -- ``None``
    when every call is done."""
    rows = list(rows)
    left = [r for r in rows if not r["done"]]
    if not left:
        return "finished", None
    return "failed", (not_done_words(left[0])
                      + (f"; {len(left)} of {len(rows)} calls not done"
                         if len(left) > 1 else ""))


def close(index: Dict[str, Any], rows) -> Dict[str, Any]:
    """THE WALK OVER -- no call left to run (:func:`still_to_run`): each
    call it never ran stamped so in the index, ``not_run`` its why -- the
    phase it needs that did not end done, or the walk that never reached
    it -- so the run's count and state say it as the status door does a
    call the ended walk never reached: failed (`run-reports.md` § 2.9), as
    a transport run's are (`transport.points_index.close`).  Asked by the
    record step that runs the finish, before it."""
    walked = set(index.get("phases") or ())
    done = {r["phase"] for r in rows if r["done"]}
    words = {r["phase"]: r["words"] for r in rows}
    for r in rows:
        if r["ended"]:
            continue
        need = next((n for n in phase(r["phase"]).needs
                     if n in walked and n not in done), None)
        entry_of(index, Call(r["phase"], r.get("name")))["not_run"] = (
            f"{words.get(need, need)} did not end done" if need
            else "the walk never reached it")
    return index


def summary(rows, phases=()) -> str:
    """The count in the run's words: *relax done · fc done · optical 7 of
    11 (4 taken over, 2 failed)* -- the calls taken over, and those that
    ended without being done, each said where there are any; the optical
    phase with no call yet, where the run walks it (``phases``, the
    index's), *after fc* -- or, the force constants done, *no mode
    picked*."""
    try:                                   # inside molbuilder
        from ..parse.dirs.job import FINAL_STATES
    except ImportError:                    # beside a job, the monitor's
        from job import FINAL_STATES  # type: ignore[no-redef]
    parts: List[str] = []
    by: Dict[str, list] = {}
    for r in rows:
        by.setdefault(r["phase"], []).append(r)
    for p in PHASES:
        got = by.get(p.name)
        if not got:
            if p.planned and by and p.name in phases:
                # BEFORE THE CALL ITS COORDINATES COME FROM, or after it
                # with no mode picked -- the selection's answer (§ 5.11 § 7).
                source_done = any(r["done"] for r in by.get(p.coordinates)
                                  or [])
                parts.append(f"{p.name}: no mode picked" if source_done
                             else f"{p.name}: after {p.coordinates}")
            continue
        n_done = sum(bool(r["done"]) for r in got)
        # TAKEN OVER is the index's ``taken_from`` (`continuation.taken_over`),
        # the key the job's rows and the status door's rows share.
        taken = sum(bool(r.get("taken_from")) and not r.get("copied_from")
                    for r in got if r["done"])
        # ENDED WITHOUT BEING DONE -- one rule for the job's rows and
        # `status`'s, each row's ``state`` its run's, as a transport run's
        # count says it (`transport.points_index.summary`).
        failed = sum(1 for r in got
                     if not r["done"] and r.get("state") in FINAL_STATES)
        if len(got) == 1:
            # ONE CALL: its word by the rule the count uses -- failed when
            # it ended without being done, whatever state its row producer
            # gave it (the job's rows and the status door's name the same
            # rule, `run-reports.md` § 2.9).
            parts.append(f"{p.name} "
                         f"{'done' if n_done else 'failed' if failed else got[0]['state']}")
        else:
            notes = ([f"{taken} taken over"] if taken else []) + (
                [f"{failed} failed"] if failed else [])
            parts.append(f"{p.name} {n_done} of {len(got)}"
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
    Raman ``1 + 2·N`` single points, ``N`` every one of the ``n_modes`` the
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
    parts.append(f"fc {_n(1 + 6 * n_free, 'SCF')} (1 + 6 x "
                 f"{_n(n_free, 'free atom')})")
    if OPTICAL in on:
        if str(cfg.raman_modes) == "all":
            calls = (f"{_n(1 + 2 * n_modes, 'single point')} (1 + 2 x "
                     f"{_n(n_modes, 'mode')})")
        else:
            calls = (f"up to {_n(1 + 2 * n_modes, 'single point')} (1 + 2 x "
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


def products_of(stage: str, label: str, base, point: Optional[Call] = None
                ) -> List[str]:
    """What a call must hold to be done: its phase's products, named on the
    label (`runfiles.compose`); ``[]`` for a relaxation, or asked with no
    call."""
    if point is None:
        return []
    return [compose(label, suf) for suf in phase(point.phase).products]


def force_constants_call(run_dir) -> Path:
    """THE FORCE-CONSTANT CALL of the run at ``run_dir`` -- its folder below
    the run (`Call.rel`): the call whose deck measured the force constants
    and which leaves the engine's `.FC` (§ 5.11 § 2), the one a reader of
    the run's force constants or of the deck they were measured with
    takes (the sweep summary, a mode's frame set)."""
    return Path(run_dir) / Call(FC, FC).rel


__all__ = ["SCHEMA", "UNIT", "RELAX", "FC", "OPTICAL", "PhaseRule", "PHASES",
           "phase", "phases_on", "IR_CHARGES", "Call", "index_path", "empty_index",
           "read_index", "write_index", "calls_of", "entry_of", "add_call",
           "rows_of", "still_to_run", "run_state", "close", "not_done_words",
           "summary", "products_of", "force_constants_call"]
