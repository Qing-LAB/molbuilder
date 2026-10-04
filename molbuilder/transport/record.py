"""The transport composite's RECORD — `archive/2026-09-01-transport-design.md` § 7,
build step P6.

TBtrans's own outputs are the truth this module reads: the k-averaged
transmission file (``<label>.TBT.AVTRANS_<L>-<R>``, two columns E vs T)
and the current line its ``.out`` prints (the binary's own Landauer
integral — parsed, never recomputed).  Both formats were pinned against
a REAL 5.4.2 run (the carbon-chain live walk, 2026-08-29; the frozen
fixtures in ``tests/data/`` are that run's files).

What lands on disk is ONE file at the calculation root,
``<label>.transport.json`` (``molbuilder/transport-result@2``): T(E)
per bias point, the I–V table -- the junction's TOTAL current, with the
figure TBtrans printed and the factor between them (:data:`CURRENT_MEANS`)
-- and the provenance that says which junction built it — the citation (from ``slot-provenance.json``) and
the atom-permutation reference, so every downstream index can be
mapped back to the relaxation's identities.

Reading is ASYNCHRONOUS by design, the same doctrine as the bench
summarizer: a point whose transmission has not run yet reads as
*pending*, never as a failure of the set — `summarize` is a reader,
and nothing is produced on a host that has produced nothing.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..atom_permutation import PERMUTATION_FILE

#: ``@2`` since 2026-10-03: ``current_a`` became the junction's TOTAL current
#: -- both spin channels -- with TBtrans's printed figure beside it
#: (``current_a_printed``).  A MAJOR bump because an ``@1`` record's
#: ``current_a`` IS the printed figure, one spin channel's, and a reader of
#: the new meaning must not read it as the total: an old record is refused by
#: its version, and `summarize run` writes it again.
TRANSPORT_RESULT_SCHEMA = "molbuilder/transport-result@2"

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

    It was an f-string until 2026-09-18, which made the docstring above false:
    `runfiles.WRITTEN` declares `.transport.json` and its module docstring says
    *"Nothing composes or splits a run-file name inline."*  Composing also
    inherits the label rule -- `compose` refuses `my.relax`, where the
    f-string built a name `runfiles.parse` cannot read back.
    """
    from ..runfiles import compose as _rf
    return Path(base_dir) / _rf(label, ".transport.json")


def parse_avtrans(text: str) -> Tuple[List[float], List[float]]:
    """``<label>.TBT.AVTRANS_<L>-<R>`` → ``(energies_ev, transmission)``.

    The format (pinned live, 5.4.2): ``#`` comment lines, then two
    columns — E in eV, and the k-averaged T(E).

    **THE OBSERVATION STANDS; ITS ATTRIBUTION WAS WRONG.**  This said E
    was relative to E_F *"when the deck said ``TS.TBT.Erange.RelToEF T``,
    which the composite's deck does"* — and that keyword is one the 5.4.2
    tbtrans cannot read (`plan.md` § 5o), so it can never have caused
    anything.  Whatever was observed in the live file was tbtrans's OWN
    default behaviour.

    That makes this docstring the best evidence bearing on § 5o's one open
    question — whether a ``%block TBT.Contour`` line's energies are
    absolute or E_F-relative — and it points at *relative by default*,
    which would mean the retired switch was never needed rather than
    merely mis-spelled.  **Recorded as evidence, not settled:** this is a
    docstring's recollection of a run whose artifact does not survive in
    the tree, and the question is closed by reading one real
    ``AVTRANS`` file beside its device ``.fdf``, not by this sentence.
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
    run used (every SIESTA deck molbuilder writes states ``Spin``, since
    2026-09-28).  No ``Spin`` line is SIESTA's own default, non-polarized."""
    from ..parse.fdf import _parse_fdf
    from ..runfiles import find_by_role
    # THE FRAMEWORK'S SEARCH for the deck (`project-layout.md` § 4.5), not a
    # glob of its suffix here (it globbed `*.fdf` until 2026-10-03).
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

    **A transport result is FIVE calculations, and the record says so.**  It
    used to describe only the last one -- the transmission points -- so a
    reader could see the deliverable or nothing, with no way to tell a run
    that had not started from one stalled at the device.  The ladder is the
    structure of the result (`engines/transport.md` § 1), and a parser that
    understands the format reports that structure rather than its final line.

    **Each rung's own key fact, which is not the same fact:**

    * *seed* / *device* -- did the SCF converge, and at what energy.  The seed
      hands the device a density; the device hands TBtrans a Hamiltonian.
    * *electrode_L* / *electrode_R* -- **the lead's Fermi level**, and this is
      the number the whole junction is referenced to: a lead is a periodic
      BULK run whose *"E_F is the reference energy"* (§ 2a.13), T(E) is
      measured relative to it, and `G = G0 * T(E_F)` is evaluated at it.  Two
      leads that disagree is a defect nothing else on the Results tab shows.
    * *transmission* -- has its own `points` / `pending` blocks already.

    **Honest, not optimistic.**  A rung with no attempt reads ``not_run``; one
    with an attempt but no `.out` reads ``no_output``; one whose parser
    refuses reads ``unreadable`` with the reason.  Nothing is inferred from a
    neighbour: `stage_inputs` makes the ladder sequential, so an unfinished
    rung explains the ones after it, but this reports what each directory
    says rather than reasoning about the order.
    """
    from ..jobset.materialize import latest_attempt, run_dir
    from ..parse import detect
    from ..runfiles import stem as rf_stem
    from .stages import STAGE_FACT, TRANSPORT_STAGES

    from ..jobset.materialize import ladder_homes
    ladder = {h.name: h.token for h in ladder_homes(base, task)}
    out: List[Dict] = []
    for name in TRANSPORT_STAGES:
        token = ladder.get(name)
        fact: Dict = {"stage": name, "token": token}
        if token is None:                      # the description omits it
            fact["state"] = "not_described"
            out.append(fact)
            continue
        # WHERE THIS RUNG RAN -- the one door's answer (`rung_containers`,
        # plan § 5w K10).  A bias SCAN puts the device and the transmission
        # under one v-dir per point (`<token>/v<V>/run-<n>`, § 4.2/4.3); a
        # look at `base / token` found no `run-<n>` there and reported a
        # FINISHED scan as `not_run` -- the transmission's until 2026-09-24,
        # the device's until K10 (the M11 review's T-F27).
        from .stages import rung_containers
        containers = [d for d, _v in rung_containers(base, task, name)]
        cand = [(c, latest_attempt(c)) for c in containers]
        cand = [(c, a) for c, a in cand if a is not None]
        if not cand:
            fact["state"] = "not_run"
            out.append(fact)
            continue
        # A SCAN'S RUNG SPEAKS FROM ITS FIRST POINT NOT FINISHED, in the
        # scan's order, and from its last once every point has -- status's
        # rule (`runstatus._job_status`), each point asked the one door
        # (`runrecord.ending`).  The newest attempt by file time spoke until
        # 2026-10-03 (plan W38 M4).
        from ..runrecord import ending
        basename = rf_stem(label, token)
        container, att = next(
            ((c, a) for c, a in cand
             if not ending(run_dir(c), basename).ok), cand[-1])
        if len(containers) > 1:
            fact["points"] = len(cand)
        fact["attempt"] = str(att.relative_to(base))
        # WHICH QUESTION THIS RUNG ANSWERS -- a column, not a branch on the
        # name (`stages.STAGE_FACT`).  A rung whose fact is its own PRODUCT
        # is not an SCF and is not asked one: TBtrans converges nothing and
        # reports no total energy, so parsing its `.out` for either could
        # only ever fail -- and did, as two hundred words of the registry's
        # format list in the cell where its state belongs.
        answers = STAGE_FACT.get(name, "scf")
        outs = _outs_newest_first(run_dir(container), token)
        # PRODUCED ANYTHING AT ALL is asked of every rung the same way, and
        # before the split below: a prepped rung that has not run reads
        # `no_output` whatever question it would have answered.
        if not outs:
            fact["state"] = "no_output"
            out.append(fact)
            continue
        if answers == "product":
            # ITS `.out` IS EVIDENCE IT RAN, NOT SOMETHING TO PARSE.  TBtrans
            # converges nothing and reports no total energy, so the state
            # comes from the run's own conclusion -- the door's answer, no
            # parser involved -- and the RESULT is the transmission this
            # record already carries in its `points` blocks.
            # THE ONE RUN-STATE DOOR, asked as every reader asks it: the
            # rung's run by its name and with its launch record (plan W38
            # M4) -- one that does not read is said, never guessed past.
            from ..parse.dirs import run_status
            from ..runrecord import LaunchRecordError, launch_record
            where = run_dir(container)
            try:
                launch = launch_record(where, basename)
            except LaunchRecordError as exc:
                fact["state"] = "unreadable"
                fact["why"] = str(exc)
                out.append(fact)
                continue
            st = run_status(where, basename, launch=launch)
            fact["state"] = "ran" if st.state == "finished" else st.state
            if st.state != "finished":
                fact["run_state"] = st.state
            fact["detail"] = st.detail
            out.append(fact)
            continue
        try:
            res = detect(str(outs[0])).parse(str(outs[0]))
        except Exception as exc:               # a refusal is an ANSWER here
            fact["state"] = "unreadable"
            # THE FIRST SENTENCE, not the essay.  `UnknownFormatError` lists
            # every registered parser on purpose -- right for someone who
            # pointed at a file and has to pick, wrong for one cell of a
            # five-row ladder, where it buried the other four.
            fact["why"] = str(exc).split(". ")[0].strip() or str(exc)
            out.append(fact)
            continue
        fact["state"] = "ran"
        fact["run_state"] = res.run_state
        fact["scf_converged"] = res.scf_converged
        frames = res.frames or []
        if frames:
            fact["energy_ev"] = frames[-1].energy
            # THE LEAD'S FERMI LEVEL, from the last SCF cycle of the last
            # frame -- the converged one.  Kept by the SIESTA parser since
            # 2026-09-18 for exactly this.  Asked by the COLUMN, so adding a
            # third lead one day is a table row and not a third name here.
            if answers == "fermi":
                hist = frames[-1].scf_history or []
                for cyc in reversed(hist):
                    if cyc.get("ef") is not None:
                        fact["fermi_ev"] = cyc["ef"]
                        break
        out.append(fact)
    return out


def collect_record(base_dir, task) -> Dict:
    """Walk the transmission attempts and build the record dict.

    Reads each point's LATEST attempt; a point with no attempt or no
    transmission output lands in ``pending`` by name.  Raises
    :class:`RecordError` only when NOTHING has run — an empty record
    would say less than the refusal.
    """
    from ..jobset.materialize import latest_attempt, run_dir, stage_home
    from .compose import PROVENANCE_FILE

    base = Path(base_dir)
    points_out: List[Dict] = []
    pending: List[Dict] = []
    token = stage_home(base, task, "transmission").token
    for v, container in _point_dirs(base, task):
        att = latest_attempt(container)   # None is the ANSWER: prepared?
        where = run_dir(container)        # ...and this is where to look
        avtrans = sorted(where.glob(f"{task.label}.TBT.AVTRANS_*"))
        if att is None or not avtrans:
            pending.append({
                "bias_v": v,
                "why": ("no attempt open" if att is None
                        else "no transmission output in "
                             f"{att.relative_to(base)}")})
            continue
        energies, trans = parse_avtrans(avtrans[0].read_text())
        spin = deck_spin(where)
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
        })
    if not points_out:
        from ..jobset.commands import block, run_first
        raise RecordError(
            "no transmission point has produced output yet -- run the "
            "transmission stage first:\n"
            + block(run_first("transmission", base=base_dir)) + "\n"
            + ("  (pending: "
               + "; ".join(f"{p['bias_v']:g} V ({p['why']})"
                           for p in pending) + ")" if pending else ""))

    provenance = None
    prov_file = base / PROVENANCE_FILE
    if prov_file.is_file():
        provenance = json.loads(prov_file.read_text())
    record: Dict = {
        "schema": TRANSPORT_RESULT_SCHEMA,
        "label": task.label,
        "energies_relative_to_ef": True,
        # THE LADDER IS THE RESULT'S STRUCTURE, so the record carries it.
        "stages": _stage_facts(base, task, task.label),
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
        "treatment": ("finite-bias" if len(points_out) + len(pending) > 1
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
        },
    }
    if pending:
        record["pending"] = pending
    return record


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
                if record.get("pending") else "")]
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
    for p in record.get("pending", ()):
        lines.append(f"  {p['bias_v']:>8.3f}  {'pending':>12}  "
                     f"{'':>12}  ({p['why']})")
    for spin, means in sorted((record.get("current_means") or {}).items()):
        lines.append(f"  {spin}: {means}")
    return "\n".join(lines)
