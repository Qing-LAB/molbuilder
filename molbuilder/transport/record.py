"""The transport composite's RECORD — `archive/2026-09-01-transport-design.md` § 7,
build step P6.

TBtrans's own outputs are the truth this module reads: the k-averaged
transmission file (``<label>.TBT.AVTRANS_<L>-<R>``, two columns E vs T)
and the current line its ``.out`` prints (the binary's own Landauer
integral — parsed, never recomputed).  Both formats were pinned against
a REAL 5.4.2 run (the carbon-chain live walk, 2026-08-29).

What lands on disk is ONE file at the calculation root,
``<label>.transport.json`` (``molbuilder/transport-result@3``): T(E)
per bias point, the I–V table -- the junction's TOTAL current, with the
figure TBtrans printed and the factor between them (:data:`CURRENT_MEANS`)
-- and the provenance that says which junction built it — the citation (from ``slot-provenance.json``) and
the atom-permutation reference, so every downstream index can be
mapped back to the relaxation's identities.

Reading is ASYNCHRONOUS by design, the same doctrine as the bench
summarizer: a point whose transmission has not run yet reads as
*pending*, never as a failure of the set, and one whose run ended without
its transmission reads as *failed*, in its run's own words — `summarize`
is a reader, and nothing is produced on a host that has produced nothing.

**COMPOSED ON READ** (`engines/transport.md` § 2a.12): every rung's state is
the one status door's (`runstatus.jobset_status`, what `jobset status` and
the Results tab's ladder say), and every point's its run's (`run_status`);
the Results tab asks `/api/transport/record`, which composes the record
each time it is opened, so a rung that ran since `summarize` is never shown
as it was.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..atom_permutation import PERMUTATION_FILE

#: ``@3``: each rung's ``state`` is the status door's word (``finished``,
#: ``failed``, ``running`` ...) with its ``detail``, and a point without a
#: transmission is ``pending`` or ``failed`` by its run's state.
#: ``current_a`` is the junction's TOTAL current -- both spin channels --
#: with TBtrans's printed figure beside it (``current_a_printed``).  An
#: older record is refused by its version, and `summarize run` writes it
#: again.
TRANSPORT_RESULT_SCHEMA = "molbuilder/transport-result@3"

#: What the record's current IS, said in the record (`engines/transport.md`
#: § 2a.4; user, 2026-10-03, Q5: "make sure the result presentation, data
#: record and the summary/comments clearly explain what is what").  TBtrans's
#: Landauer integral is one spin channel's -- I = (e/h)∫T, no factor 2 for
#: spin (`m_tbt_save.F90`) -- while the conductance beside it is in
#: G0 = 2e²/h, both channels.
CURRENT_MEANS = {
    "non-polarized": (
        "current_a is the junction's total current, both spin channels: "
        "TBtrans prints one spin channel's (its Landauer integral has no "
        "factor 2 for spin), and the two channels of a non-polarized "
        "calculation carry the same current, so the total is twice the "
        "printed figure (current_a_printed, as TBtrans printed it)."),
    "polarized": (
        "a spin-polarized junction's total current is the sum of its two "
        "channels, which this record reads once both are (plan K21); until "
        "then current_a is empty and current_a_printed is the one channel "
        "TBtrans printed first."),
}

#: THE CAVEAT that goes with any DFT-NEGF conductance (`engines/transport.md`
#: § 1 and § 2a.12), carried on the result so every reader says it.
CAVEAT = (
    "A DFT-NEGF conductance with a plain GGA functional places the "
    "molecule's levels too close to the Fermi level, and so overestimates a "
    "molecular junction's conductance, often by one to two orders of "
    "magnitude.  Read T(E) for where the levels are and how they couple, "
    "not for an absolute conductance.")

#: The SCF row fields the report draws (`web/results.md` § 2.5) -- the
#: SIESTA reader's own names; a NEGF row carries its charge too.
_SCF_FIELDS = ("cycle", "energy", "delta_E", "dDmax", "dHmax", "ef",
               "phase", "dq", "vha_ev", "charges")

#: TBtrans prints the Landauer current as its own integral -- one line
#: per electrode pair.  Matched loosely on the unit scaffold so custom
#: electrode names still parse; the numbers are Fortran-shaped
#: (``0.309835E-04``, ``-.619664E-05``), which ``float`` accepts.
_CURRENT_RE = re.compile(
    r"V \[V\] / I \[A\]:\s*(\S+)\s*V\s*/\s*(\S+)\s*A")


class RecordError(Exception):
    """The record cannot be built — the message names what to run or
    fix first, ready to surface verbatim."""


def record_path(base_dir, label: str) -> Path:
    """The ONE spelling of the record's location -- composed, not concatenated.

    `runfiles.WRITTEN` declares `.transport.json` and its module docstring says
    *"Nothing composes or splits a run-file name inline."*  Composing also
    inherits the label rule -- `compose` refuses `my.relax`, a name
    `runfiles.parse` cannot read back.
    """
    from ..runfiles import compose as _rf
    return Path(base_dir) / _rf(label, ".transport.json")


def parse_avtrans(text: str) -> Tuple[List[float], List[float]]:
    """``<label>.TBT.AVTRANS_<L>-<R>`` → ``(energies_ev, transmission)``.

    The format (pinned live, 5.4.2): ``#`` comment lines, then two
    columns — E in eV, and the k-averaged T(E).

    E is read as relative to E_F, tbtrans's OWN default behaviour as a live
    file showed it -- the evidence bearing on `plan.md` § 5o's open
    question, whether a ``%block TBT.Contour`` line's energies are absolute
    or E_F-relative.  **Recorded as evidence, not settled:** the question is
    closed by reading one real ``AVTRANS`` file beside its device ``.fdf``.
    """
    energies: List[float] = []
    trans: List[float] = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 2:
            continue
        try:
            e, t = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        energies.append(e)
        trans.append(t)
    if not energies:
        raise RecordError("no transmission rows parsed -- not an "
                          "AVTRANS file?")
    return energies, trans


def deck_spin(run_dir) -> str:
    """The spin the run's own deck states -- ``non-polarized`` or
    ``polarized`` -- read from the deck in its run directory, the input the
    run used (every SIESTA deck molbuilder writes states ``Spin``).  No ``Spin`` line is SIESTA's own default, non-polarized."""
    from ..parse.fdf import _parse_fdf
    from ..runfiles import find_by_role
    # THE FRAMEWORK'S SEARCH for the deck (`project-layout.md` § 4.5), not a
    # glob of its suffix here.
    for deck in find_by_role(run_dir, ".fdf"):
        scalars, _blocks = _parse_fdf(deck.read_text(encoding="utf-8",
                                                     errors="replace"))
        said = (scalars.get("spin") or [None])[0]
        if said:
            return ("polarized" if str(said).strip().lower() == "polarized"
                    else "non-polarized")
    return "non-polarized"


def total_current(printed: Optional[float], spin: str) -> Optional[float]:
    """The junction's total current from TBtrans's printed one-channel
    figure (:data:`CURRENT_MEANS`): twice it, non-polarized; a polarized
    junction's total is its two channels' sum (plan K21), so ``None`` here."""
    if printed is None or spin != "non-polarized":
        return None
    return 2.0 * printed


def parse_current_a(out_text: str) -> Optional[float]:
    """The current (amps) from TBtrans's own ``.out`` line, or ``None``
    when the run printed none (an equilibrium-only window prints
    I = 0, which parses as the honest 0.0)."""
    m = _CURRENT_RE.search(out_text)
    if not m:
        return None
    try:
        return float(m.group(2))
    except ValueError:
        return None


def conductance_g0(energies: List[float], trans: List[float]
                   ) -> Optional[float]:
    """T interpolated at E = 0 (the grid is E − E_F) — G(E_F) in G0."""
    import numpy as np
    e = np.asarray(energies, dtype=float)
    t = np.asarray(trans, dtype=float)
    if e.min() > 0 or e.max() < 0:
        return None                    # window does not straddle E_F
    return float(np.interp(0.0, e, t))


def _outs_newest_first(where: Path, token: str) -> List[Path]:
    """A rung's engine outputs in ``where``, its NEWEST RUN first -- the
    run door's (`runs.Run.outputs`), by the run number every one carries,
    never by file time (plan W38 M4): a copy, a touch or a restored folder
    reorders the times and not the runs.  ``[]`` where no run of ours is."""
    from ..runs import run_of
    run = run_of(where, stage=token)
    return run.outputs if run is not None else []


def _point_dirs(base: Path, task) -> List[Tuple[float, Path]]:
    """``(voltage, transmission point container)`` per § 4.2/4.3: the one
    door's folders (`stages.rung_containers`), each with the bias it ran
    at -- a single-bias calculation's one folder at its one point."""
    from .stages import rung_containers
    v0 = float(task.bias[0]) if getattr(task, "bias", ()) else 0.0
    return [(v0 if v is None else v, d)
            for d, v in rung_containers(base, task, "transmission")]


def _stage_facts(base: Path, task, label: str) -> List[Dict]:
    """One entry per rung of the ladder: where it stands, and its own answer.

    **A transport result is FIVE calculations, and the record says so**, so
    a reader can tell a run that has not started from one stalled at the
    device (`engines/transport.md` § 1).

    **WHERE EACH RUNG STANDS is the one status door's answer**
    (`runstatus.jobset_status` -- what `jobset status` prints and the
    Results tab's ladder draws): its ``state`` and ``detail`` in that
    door's words, a bias scan's rung speaking from its first point not
    finished.  Nothing here reads a run's state a second way.

    **Each rung's own key fact, which is not the same fact**, read from the
    run's engine output once it has one:

    * *seed* / *device* -- did the SCF converge, and at what energy.  The seed
      hands the device a density; the device hands TBtrans a Hamiltonian.
    * *electrode_L* / *electrode_R* -- **the lead's Fermi level**, and this is
      the number the whole junction is referenced to: a lead is a periodic
      BULK run whose *"E_F is the reference energy"* (§ 2a.13), T(E) is
      measured relative to it, and `G = G0 * T(E_F)` is evaluated at it.  Two
      leads that disagree is a defect nothing else on the Results tab shows.
    * *transmission* -- has its own `points` / `pending` / `failed` blocks.
    """
    from ..jobset.materialize import ladder_homes
    from ..jobset.model import FILENAME as JOBSET_FILENAME, JobSet
    from ..jobset.runstatus import jobset_status
    from .stages import STAGE_FACT, TRANSPORT_STAGES, rung_containers

    jpath = base / JOBSET_FILENAME
    status = jobset_status(JobSet.load(jpath) if jpath.is_file() else None,
                           base)
    by_name = {s.name: s for s in status.stages}
    tokens = {h.name: h.token for h in ladder_homes(base, task)}
    out: List[Dict] = []
    for name in TRANSPORT_STAGES:
        s = by_name.get(name)
        if s is None:                          # the description omits it
            out.append({"stage": name, "token": None,
                        "state": "not_described"})
            continue
        token = tokens.get(name)
        fact: Dict = {"stage": name, "token": token, "state": s.state,
                      "detail": s.detail}
        if s.attempt and s.dir:
            fact["attempt"] = f"{s.dir}/{s.attempt}"
        # WHICH QUESTION THIS RUNG ANSWERS -- a column, not a branch on the
        # name (`stages.STAGE_FACT`).  The transmission's answer is its
        # points; TBtrans converges nothing and reports no total energy.
        answers = STAGE_FACT.get(name, "scf")
        outs = (_outs_newest_first(base / s.dir / s.attempt, token)
                if answers != "product" and s.attempt and s.dir else [])
        if not outs:
            out.append(fact)
            continue
        fact.update(_science(outs[0], answers))
        if len(rung_containers(base, task, name)) > 1:
            # A SCAN'S RUNG, POINT BY POINT -- the report's convergence
            # card follows the selected bias (`web/results.md` § 2.5).
            fact["by_point"] = _rung_points(base, task, name, token, answers)
        out.append(fact)
    return out


def _science(out: Path, answers: str) -> Dict:
    """A rung run's own answer, read from its engine output: its SCF as it
    ran (every row of its last step, phase-tagged, with each phase's
    criteria and how it ended), whether it converged and at what energy,
    a NEGF phase's own figures, a lead's E_F -- or ``unreadable`` with the
    parser's first sentence."""
    from ..parse import detect
    fact: Dict = {}
    try:
        res = detect(str(out)).parse(str(out))
    except Exception as exc:                   # a refusal is an ANSWER here
        # THE FIRST SENTENCE, not the essay: `UnknownFormatError` lists
        # every registered parser, which buries the other four rungs.
        fact["unreadable"] = str(exc).split(". ")[0].strip() or str(exc)
        return fact
    fact["scf_converged"] = res.scf_converged
    frames = res.frames or []
    info = res.runtime_info or {}
    # THE RUN'S SCF, AS IT RAN (`web/results.md` § 2.5): every row of the
    # last step, phase-tagged -- a device's periodic start and its NEGF
    # loop apart -- with what each phase had to reach and how it ended.
    hist = (frames[-1].scf_history or []) if frames else []
    fact["scf"] = [{k: c[k] for k in _SCF_FIELDS if k in c} for c in hist]
    if info.get("scf_criteria"):
        fact["scf_criteria"] = info["scf_criteria"]
    if info.get("scf_phases"):
        fact["scf_phases"] = info["scf_phases"]
    # THE NEGF PHASE'S OWN FIGURES -- never the periodic start's
    # (`engines/transport.md` § 2a.12): its last row's E_F and charge, and
    # how many cycles it ran.
    negf = [c for c in hist if c.get("phase") == "negf"]
    if negf:
        last = negf[-1]
        fact["negf"] = {"cycles": len(negf),
                        **{k: last[k] for k in ("ef", "dq", "charges",
                                                "energy") if k in last}}
    if frames:
        fact["energy_ev"] = frames[-1].energy
        # THE LEAD'S FERMI LEVEL, from the last SCF cycle of the last frame
        # -- the converged one.  Asked by the COLUMN, so adding a third lead
        # one day is a table row and not a third name here.
        if answers == "fermi":
            for cyc in reversed(hist):
                if cyc.get("ef") is not None:
                    fact["fermi_ev"] = cyc["ef"]
                    break
    return fact


def _rung_points(base: Path, task, name: str, token: str,
                 answers: str) -> List[Dict]:
    """``[{bias_v, attempt, state, detail, ...science}]`` -- each bias
    point of a scan's rung, its state the run door's (`run_status`) and its
    own answer (:func:`_science`)."""
    from ..jobset.materialize import latest_attempt, run_dir
    from ..parse.dirs import run_status
    from ..runfiles import RunNames
    from ..runrecord import LaunchRecordError, launch_record
    from .stages import rung_containers
    names = RunNames.of(task.label, token, task.shape)
    out: List[Dict] = []
    for d, v in rung_containers(base, task, name):
        att = latest_attempt(d)
        entry: Dict = {"bias_v": v}
        if att is None:
            entry.update(state="not-started", detail="not prepped")
            out.append(entry)
            continue
        where = run_dir(d)
        entry["attempt"] = str(att.relative_to(base))
        try:
            st = run_status(where, names.stem,
                            launch=launch_record(where, names))
            entry.update(state=st.state, detail=st.detail)
        except LaunchRecordError as exc:
            entry.update(state="unreadable", detail=str(exc))
        outs = _outs_newest_first(where, token)
        if outs:
            entry.update(_science(outs[0], answers))
        out.append(entry)
    return out


def device_regions(base: Path) -> Dict[str, List[int]]:
    """The device's regions molbuilder owns -- L-electrode, bridge,
    R-electrode -- as atom indices (0-based, the composed junction's order,
    which is every rung's deck's), read from the calculation's composed
    junction (`compose`'s copy and its sidecar).  ``{}`` before it is
    composed."""
    from ..runfiles import JUNCTION_FILE
    from ..workingcopy_structure import StructureCodec
    from .sort import ELECTRODE_LABELS, REGION_BRIDGE
    path = Path(base) / JUNCTION_FILE
    if not path.is_file():
        return {}
    regions = StructureCodec().load(path).regions or {}
    return {label: [int(i) for i in regions.get(label, ())]
            for label in (ELECTRODE_LABELS[0], REGION_BRIDGE,
                          ELECTRODE_LABELS[1])
            if regions.get(label)}


def _chain(base: Path, stages: List[Dict]) -> List[Dict]:
    """``[{stage, attempt, gathered: [{file, from}]}]`` -- what each rung's
    attempt was gathered from, read from its `.gathered-from`
    (`runrecord.read_gathered_from`); a rung that gathered nothing is
    listed with ``[]``."""
    from ..runrecord import read_gathered_from
    return [{"stage": st["stage"], "attempt": st["attempt"],
             "gathered": read_gathered_from(base / st["attempt"])}
            for st in stages if st.get("attempt")]


def collect_record(base_dir, task, *, partial: bool = False) -> Dict:
    """Walk the transmission attempts and build the record dict.

    Reads each point's LATEST attempt, its state the run door's
    (`parse.dirs.run_status`): a point with its transmission is in
    ``points``; one not launched, queued or running is ``pending`` -- never
    a failure; one whose run failed, stopped, or ended without its
    transmission is ``failed``, saying which (the displacement sweep's
    split, `spectra.displacement_sweep`).  Raises :class:`RecordError` when
    no point has its transmission -- an empty record would say less than
    the refusal -- unless ``partial``: the Results tab's read, where a
    ladder in progress has a report (`engines/transport.md` § 2a.12).
    """
    from ..jobset.materialize import latest_attempt, run_dir, stage_home
    from ..parse.dirs import run_status
    from ..runfiles import RunNames
    from ..runrecord import LaunchRecordError, launch_record
    from .compose import PROVENANCE_FILE
    from .tbtnc import (ORBITAL_NOTE, ORBITAL_TYPES, TbtError, point_dos,
                        tbt_file)

    base = Path(base_dir)
    points_out: List[Dict] = []
    pending: List[Dict] = []
    failed: List[Dict] = []
    token = stage_home(base, task, "transmission").token
    names = RunNames.of(task.label, token, task.shape)
    regions = device_regions(base)
    opened = False                        # an attempt open: it is prepped
    for v, container in _point_dirs(base, task):
        att = latest_attempt(container)   # None is the ANSWER: prepared?
        opened = opened or att is not None
        if att is None:
            pending.append({"bias_v": v, "state": "not-started",
                            "why": "no attempt opened yet -- prep and "
                                   "launch the transmission"})
            continue
        where = run_dir(container)        # ...and this is where to look
        rel = str(att.relative_to(base))
        # THE POINT'S STATE, the run door's -- asked as every reader asks
        # it, with its launch record; one that does not read is said.
        try:
            launch = launch_record(where, names)
        except LaunchRecordError as exc:
            failed.append({"bias_v": v, "attempt": rel,
                           "state": "unreadable", "why": str(exc)})
            continue
        st = run_status(where, names.stem, launch=launch)
        avtrans = sorted(where.glob(f"{task.label}.TBT.AVTRANS_*"))
        if st.state != "finished" or not avtrans:
            entry = {"bias_v": v, "attempt": rel, "state": st.state,
                     "why": st.detail}
            if st.state in ("pending", "queued", "running"):
                pending.append(entry)
            else:
                if st.state == "finished":
                    entry["why"] = (f"the run finished without writing its "
                                    f"transmission ({st.detail})")
                failed.append(entry)
            continue
        energies, trans = parse_avtrans(avtrans[0].read_text())
        spin = deck_spin(where)
        # THE DOS, ITS PARTS AND THE EIGENCHANNELS, from the point's own
        # `.TBT.nc` (`tbtnc.point_dos`; `web/results.md` § 2.5).
        nc = tbt_file(where, task.label)
        dos = None
        dos_why = None
        if nc is None:
            dos_why = f"{task.label}.TBT.nc is not in {rel}"
        else:
            try:
                dos = point_dos(nc, regions)
            except TbtError as exc:
                dos_why = str(exc)
        current = None
        # THE NEWEST RUN'S, by its number (`_outs_newest_first`).
        for out in _outs_newest_first(where, token):
            current = parse_current_a(out.read_text())
            if current is not None:
                break
        points_out.append({
            "bias_v": v,
            "attempt": str(att.relative_to(base)),
            "transmission_file": avtrans[0].name,
            "energy_ev": energies,
            "transmission": trans,
            "conductance_g0": conductance_g0(energies, trans),
            # THE TOTAL, and the figure it came from (`CURRENT_MEANS`).
            "current_a": total_current(current, spin),
            "current_a_printed": current,
            "spin": spin,
            **({"dos": dos} if dos is not None else {"dos_why": dos_why}),
        })
    if not points_out and not partial:
        # THE WAY ON, by what the stage's state says: prepped (an attempt is
        # open), it is launched or let finish -- a prepped stage refuses a
        # second prep (`job-system.md` § 5.0); else prepped, then launched.
        from ..jobset.commands import block, launch_lines, run_first
        raise RecordError(
            "no transmission point has its transmission yet -- "
            + ("launch the transmission stage, or let it finish:\n"
               + block(launch_lines("run", "transmission", base=base_dir))
               if opened else
               "run the transmission stage first:\n"
               + block(run_first("transmission", base=base_dir)))
            + "\n"
            + "".join(f"  ({what}: "
                      + "; ".join(f"{p['bias_v']:g} V, {p['state']} "
                                  f"({p['why']})" for p in got) + ")\n"
                      for what, got in (("pending", pending),
                                        ("failed", failed)) if got))

    stages = _stage_facts(base, task, task.label)
    provenance = None
    prov_file = base / PROVENANCE_FILE
    if prov_file.is_file():
        provenance = json.loads(prov_file.read_text())
    record: Dict = {
        "schema": TRANSPORT_RESULT_SCHEMA,
        "label": task.label,
        "energies_relative_to_ef": True,
        # THE LADDER IS THE RESULT'S STRUCTURE, so the record carries it.
        "stages": stages,
        # WHAT THE RESULT IS ENTITLED TO BE CALLED (§ 2a.10, ruled
        # 2026-09-16).  The mechanism is identical either way; what differs is
        # how many device SCFs were paid for, and therefore what may be
        # claimed.  One slice is T(E), and an I-V read off it is the
        # LINEAR-RESPONSE approximation -- integrating a zero-bias curve
        # cannot reproduce a resonance entering the window, nor that
        # resonance moving under the field.  A re-converged scan is T(E, V)
        # and its I-V carries no such caveat.
        #
        # Recorded here, not decided in the browser: § 2a.12 requires the
        # treatment NAMED BESIDE THE CURVE and not in metadata, because "the
        # two kinds of I-V are different claims and look identical on a plot".
        # THE DESCRIPTION'S, as stated (`task.bias_treatment`, TD7): a list
        # of several voltages states `low-bias` or `re-converged`; one is a
        # single bias.  Never inferred from how many points ran.
        "treatment": (task.bias_treatment if len(task.bias) > 1
                      else "single-bias"),
        "points": points_out,
        "iv": {
            "voltages_v": [p["bias_v"] for p in points_out],
            "current_a": [p["current_a"] for p in points_out],
            "current_a_printed": [p["current_a_printed"] for p in points_out],
        },
        # WHAT THE CURRENT IS, in the record's own words -- for each spin the
        # points were run with (one, for a junction decided once).
        "current_means": {s: CURRENT_MEANS[s]
                          for s in sorted({p["spin"] for p in points_out})},
        "provenance": {
            "slot": provenance,
            "atom_permutation": PERMUTATION_FILE,
            # WHAT EACH RUNG WAS GATHERED FROM -- its attempt's
            # `.gathered-from`, never the newest attempt by file time
            # (`engines/transport.md` § 2a.12).
            "chain": _chain(base, stages),
        },
        "caveat": CAVEAT,
        # THE ORBITAL MENU of the PDOS of a selection, and what it cannot
        # form (`tbtnc`; `web/results.md` § 2.5).
        "pdos_orbitals": {"types": ["all", *ORBITAL_TYPES],
                          "note": ORBITAL_NOTE},
    }
    # EVERY KEY, EVERY RECORD: an empty list where there is none.
    record["pending"] = pending
    record["failed"] = failed
    return record


def selection_pdos(base_dir, task, bias_v: float, atoms, orbitals: str
                   ) -> Dict:
    """The PDOS of ``atoms`` at the bias point ``bias_v``, narrowed to one
    orbital type (`tbtnc.selection_pdos`; `web/results.md` § 2.5): read from
    that point's transmission run's ``.TBT.nc``, each orbital's type from the
    device run's ``.ORB_INDX`` at the same point -- the two runs share the
    composed junction's atom order.  :class:`RecordError` names what is
    missing."""
    from ..jobset.materialize import run_dir
    from .stages import rung_containers
    from .tbtnc import TbtError, selection_pdos as _pdos, tbt_file
    base = Path(base_dir)

    def _at(stage: str) -> Optional[Path]:
        v0 = float(task.bias[0]) if getattr(task, "bias", ()) else 0.0
        for d, v in rung_containers(base, task, stage):
            if abs((v0 if v is None else v) - float(bias_v)) < 1e-9:
                return run_dir(d)
        return None

    where = _at("transmission")
    if where is None:
        raise RecordError(f"no transmission point at {bias_v:g} V")
    nc = tbt_file(where, task.label)
    if nc is None:
        raise RecordError(f"the transmission at {bias_v:g} V has written "
                          f"no {task.label}.TBT.nc")
    dev = _at("device")
    orb = (dev / f"{task.label}.ORB_INDX") if dev is not None else None
    try:
        return _pdos(nc, orb, atoms, orbitals)
    except TbtError as exc:
        raise RecordError(str(exc)) from exc


def write_record(base_dir, record: Dict) -> Path:
    from ..persist import write_json
    out = record_path(base_dir, record["label"])
    write_json(out, record)
    return out


def iv_table_text(record: Dict) -> str:
    """The printed deliverable: one row per point — G(E_F), the junction's
    total current, and the figure TBtrans printed -- then what the current
    is, in the record's words (:data:`CURRENT_MEANS`)."""
    lines = [f"transport record — {record['label']}: "
             f"{len(record['points'])} point(s)"
             + (f", {len(record['pending'])} pending"
                if record.get("pending") else "")
             + (f", {len(record['failed'])} failed"
                if record.get("failed") else "")]
    lines.append(f"  {'V [V]':>8}  {'G(E_F) [G0]':>12}  "
                 f"{'I total [A]':>12}  {'I printed [A]':>13}")
    for p in record["points"]:
        g = p["conductance_g0"]
        i, i0 = p["current_a"], p.get("current_a_printed")
        lines.append(
            f"  {p['bias_v']:>8.3f}  "
            + (f"{g:>12.4f}" if g is not None else f"{'--':>12}")
            + "  "
            + (f"{i:>12.4e}" if i is not None else f"{'--':>12}")
            + "  "
            + (f"{i0:>13.4e}" if i0 is not None else f"{'--':>13}"))
    for what in ("pending", "failed"):
        for p in record.get(what, ()):
            lines.append(f"  {p['bias_v']:>8.3f}  {what:>12}  "
                         f"{p['state']:>12}  ({p['why']})")
    for spin, means in sorted((record.get("current_means") or {}).items()):
        lines.append(f"  {spin}: {means}")
    return "\n".join(lines)
