"""A SIESTA VIBRATION AS ONE RUN IN PHASES -- the phases, each SIESTA call
a point of the run, and the run's index (`engines/vibration.md` § 5.11).

MODULE  siesta.vibration_phases (floor 1 + stdlib; travels in
        ``mb_vibration.pyz``, `runwrap.VIBRATION_COMPANIONS`)
ROLE    the one table of the phases (:data:`PHASES`): what each phase's
        deck is, what turns it on, where its calls' coordinates and seed
        come from, what a call must hold to be done, which warm-files
        section it reads, which phases it needs -- read by prep (the decks
        and run scripts), launch (the walk, the calls it opens and takes
        over), the run's own python (the plan, record and finish verbs)
        and every reader of the run's state.  The index
        (``<label>_<stage>-run<N>.phases.json``, :func:`write_index` /
        :func:`read_index`) lists every call: what it is, from what, how it
        ended -- written by launch for the calls it opens and by the run's
        own python for the rest, rewritten whole.  :data:`POINTS` is the
        kind's declaration as a rung of points (`jobset.points`).
USED-BY `jobset.prep`, `jobset.submit`, `jobset.points`,
        `spectra.siesta_vibration`, `parse.sidecars.vibration_phases`
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# TWO WAYS, because this module travels beside a job (`runwrap.
# VIBRATION_COMPANIONS`), where the package is not installed.
try:                                       # inside molbuilder
    from ..persist import check_schema, json_text, read_json, write_bytes
    from ..runfiles import FIRST_ATTEMPT, ROLE_PHASES, ROLE_SPECTRA, compose
except ImportError:                        # beside a job
    from persist import (check_schema, json_text,  # type: ignore[no-redef]
                         read_json, write_bytes)
    from runfiles import FIRST_ATTEMPT, ROLE_PHASES, ROLE_SPECTRA, compose

SCHEMA = "molbuilder/vibration-phases@1"

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
    #: vocabulary (`paths.KIND_ROLES["vibration"]`, § 5.11 § 9): the one
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
              warm_kind="optimization", needs=()),
    PhaseRule(FC, part="fc", role="force_constants", switch=None,
              coordinates=RELAX, seed=(RELAX, ".DM"), products=(".FC",),
              warm_kind="vibration", needs=(RELAX,)),
    PhaseRule(OPTICAL, part="optical", role="optical",
              switch=("es_mode_selection", "skip"),
              coordinates=FC, seed=(FC, ".DM"), products=(".EPSIMG",),
              warm_kind="optical", needs=(FC,), planned=True),
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

def index_path(run_dir, names, run: int) -> Path:
    """Where the run's index lies: ``<label>_<stage>-run<N>.phases.json`` at
    the run's root (`runfiles.ROLE_PHASES`)."""
    return Path(run_dir) / names.name(ROLE_PHASES, run=run)


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
    schema."""
    p = Path(path)
    try:
        d = read_json(p)
        check_schema(d.get("schema") if isinstance(d, dict) else "",
                     SCHEMA, label=p.name)
    except (OSError, ValueError) as exc:
        raise ValueError(f"{p.name}: {exc}") from exc
    return d


def write_index(path, index: Dict[str, Any]) -> Path:
    """The index written whole, atomically (`persist.write_bytes`)."""
    p = Path(path)
    write_bytes(p, json_text(index).encode("utf-8"))
    return p


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
#  The kind's declaration as a rung of points (`jobset.points`)          #
# --------------------------------------------------------------------- #

def summary(rows, phases=()) -> str:
    """The count in the run's words: *relax done · fc done · optical 7 of
    11 (4 taken over)*; the optical phase with no call yet, where the run
    walks it (``phases``, the index's), *after fc* -- or, the force
    constants done, *no mode picked*."""
    parts: List[str] = []
    by: Dict[str, list] = {}
    for r in rows:
        by.setdefault(r["phase"], []).append(r)
    for p in PHASES:
        got = by.get(p.name)
        if not got:
            if p.name == OPTICAL and by and OPTICAL in phases:
                # BEFORE THE FORCE CONSTANTS, or after them with no mode
                # picked -- the selection's answer (§ 5.11 § 7).
                fc_done = any(r["done"] for r in by.get(FC) or [])
                parts.append("optical: no mode picked" if fc_done
                             else "optical: after fc")
            continue
        n_done = sum(bool(r["done"]) for r in got)
        taken = sum(bool(r.get("started_from")) and not r.get("copied_from")
                    for r in got if r["done"])
        if len(got) == 1:
            parts.append(f"{p.name} {'done' if n_done else got[0]['state']}")
        else:
            parts.append(f"{p.name} {n_done} of {len(got)}"
                         + (f" ({taken} taken over)" if taken else ""))
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
    Raman ``1 + 2·N`` single points, ``N`` the listed modes under
    ``explicit`` that are among the ``n_modes`` -- none at all for an
    empty list -- at most ``n_modes``
    under ``all``, the modes the run reports (prep's count, the rank rule,
    `validation.spectra.rigid_motion_count`); each ``.EPSIMG``'s size from
    its photon energies (``Optical.Energy.Maximum`` over a step of
    ``Optical.Broaden / 10``, `Src/optical.F`).  Only the phases the
    settings turn on (:func:`phases_on`).  The wall stays the person's:
    nothing here judges it."""
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
        calls = None
        if str(cfg.es_mode_selection) == "explicit":
            from ..spectra.selection import SelectionError, parse_index_list
            try:
                listed = parse_index_list(str(cfg.es_explicit_indices or ""),
                                          first=1)
            except SelectionError:
                parts.append("optical: the explicit mode list does not read "
                             "-- prep's check says why")
                listed = None
            # A LISTED NUMBER ABOVE THE MODES THE RUN REPORTS names none:
            # the run drops it (§ 4.8), so it is counted out here.
            n = (None if listed is None
                 else sum(1 for k in listed if 1 <= k <= n_modes))
            if listed is not None and not listed:
                parts.append("optical: no mode listed, so no optical call")
            elif n == 0:
                parts.append(f"optical: no listed mode is one of the "
                             f"{_n(n_modes, 'mode')} the run reports, so no "
                             f"optical call")
            elif n is not None:
                beyond = len(listed) - n
                calls = (f"{_n(1 + 2 * n, 'single point')} (1 + 2 x "
                         f"{_n(n, 'mode')}"
                         + (f"; {beyond} listed above the run's {n_modes} "
                            f"dropped" if beyond else "") + ")")
        else:
            calls = (f"up to {_n(1 + 2 * n_modes, 'single point')} (1 + 2 x "
                     f"the modes its window keeps, at most {n_modes})")
        if calls is not None:
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


def _job_phases(calc, stage: str) -> Tuple[str, ...]:
    """The phases the stage's job was prepared with, as its row in the
    calculation's plan records them (`model.Job.phases`, written by prep)
    -- the plan as ``calc`` holds it: a prep's, with its rows merged in,
    else the one on disk; ``()`` for a stage prepared without phases, or
    none prepared."""
    job = calc.job(stage)
    return tuple(getattr(job, "phases", ()) or ())


def _declared_rung_points(calc, stage: str, *, read=None
                          ) -> Tuple[Call, ...]:
    """The calls known before the job -- one each for the relaxation and
    the force constants the stage runs (:func:`_job_phases`); the optical
    calls exist only once the force constants have run."""
    return tuple(Call(p, p) for p in _job_phases(calc, stage)
                 if p in (RELAX, FC))


def _declared_points_in(run, task, stage: str, base) -> List[Tuple[Path, Call]]:
    """A run's calls -- those its index lists (:func:`read_index`), each
    with its folder below the run; before launch wrote the index, the calls
    prep opened, the plan's row's phases (`Job.phases`).  The index is the
    run's one, written at its first launch (a run folder holds one run of
    points; a call's retries are the call's own)."""
    from ..jobset.naming import stage_home
    run = Path(run)
    names = stage_home(base, task, stage).names(task.shape)
    path = index_path(run, names, FIRST_ATTEMPT)
    if not path.is_file():
        # BEFORE LAUNCH WROTE THE INDEX: the calls prep opened -- the
        # plan's row's phases (`Job.phases`), as a transport rung lists its
        # prepared points.
        from ..jobset.model import JobSet
        plan = JobSet.of(base)
        job = plan.job(stage) if plan is not None else None
        return [(run / Call(p, p).rel, Call(p, p))
                for p in (getattr(job, "phases", None) or ())
                if p in (RELAX, FC)]
    return [(run / c.rel, c) for c in calls_of(read_index(path))]


def _declared_hands_on(stage: str, label: str, base) -> list:
    return []


def _declared_containers(calc, stage: str) -> Tuple[str, ...]:
    """The folders a run holds besides its calls, opened with the run
    (§ 5.11 § 2): the optical calls' container, when the stage's job runs
    the optical phase -- the plan step puts each call below it."""
    return (OPTICAL,) if OPTICAL in _job_phases(calc, stage) else ()


def _declared_summary(rows, run, names) -> str:
    """The count's words for `status` (`points.PointsOf.summary`): the
    phases the run walks read off its index."""
    path = index_path(run, names, FIRST_ATTEMPT)
    try:
        phases = read_index(path).get("phases") or ()
    except (OSError, ValueError):
        phases = ()
    return summary(rows, phases)


def _also_needs(run, names, task, stage: str, base) -> Optional[str]:
    """What the run needs beyond its calls done: its finish ended with exit
    0 (the index's entry) and its result on disk (§ 5.11 § 7)."""
    path = index_path(run, names, FIRST_ATTEMPT)
    if not path.is_file():
        return "no index of its calls"
    fin = (read_index(path).get("finish") or {})
    if fin.get("ended") is None:
        return "the finish did not end"
    if fin.get("exit") not in (0, "0"):
        return f"the finish exited {fin.get('exit')}: no result"
    from ..runs import on_label_at
    if on_label_at(run, ROLE_SPECTRA) is None:
        return "the finish left no result at the run's root"
    return None


try:
    from ..jobset.points import PointsOf
except ImportError:                        # beside a job: the declaration is this machine's
    PointsOf = None

#: THE DECLARATION (`jobset.points.DECLARED`): a SIESTA vibration run's
#: points are its SIESTA calls -- the relaxation's and the force constants'
#: known at launch, the optical ones planned in the job.
POINTS = (PointsOf(kind="vibration",
                   rung_points=_declared_rung_points,
                   points_in=_declared_points_in,
                   products_of=products_of,
                   hands_on=_declared_hands_on,
                   summary=_declared_summary,
                   converges=lambda stage: True,
                   verdict_counts=lambda stage, point: point.phase != RELAX,
                   also_needs=_also_needs,
                   containers=_declared_containers,
                   planned_in_job=True, one_wall=True, unit="call")
          if PointsOf is not None else None)


__all__ = ["SCHEMA", "RELAX", "FC", "OPTICAL", "PhaseRule", "PHASES",
           "phase", "phases_on", "Call", "index_path", "empty_index",
           "read_index", "write_index", "calls_of", "entry_of", "add_call",
           "summary", "products_of", "force_constants_call", "POINTS"]
