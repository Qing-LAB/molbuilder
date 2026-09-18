"""The transport composite's RECORD — `archive/2026-09-01-transport-design.md` § 7,
build step P6.

TBtrans's own outputs are the truth this module reads: the k-averaged
transmission file (``<label>.TBT.AVTRANS_<L>-<R>``, two columns E vs T)
and the current line its ``.out`` prints (the binary's own Landauer
integral — parsed, never recomputed).  Both formats were pinned against
a REAL 5.4.2 run (the carbon-chain live walk, 2026-08-29; the frozen
fixtures in ``tests/data/`` are that run's files).

What lands on disk is ONE file at the calculation root,
``<label>.transport.json`` (``molbuilder/transport-result@1``): T(E)
per bias point, the I–V table, and the provenance that says which
junction built it — the citation (from ``slot-provenance.json``) and
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

TRANSPORT_RESULT_SCHEMA = "molbuilder/transport-result@1"

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
    """The ONE spelling of the record's location."""
    return Path(base_dir) / f"{label}.transport.json"


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


def _point_dirs(base: Path, task) -> List[Tuple[float, Path]]:
    """``(voltage, transmission point container)`` per § 4.2/4.3: the
    stage dir itself for a single point, one v-dir each for a scan."""
    from ..identity import StageRef
    from .stages import bias_points, bias_token
    token = next(r.token for r in
                 StageRef.ladder([s.name for s in task.stages])
                 if r.name == "transmission")
    stage_dir = base / token
    points = bias_points(task)
    if not points:
        v0 = (task.bias[0] if getattr(task, "bias", ()) else 0.0)
        return [(float(v0), stage_dir)]
    return [(float(v), stage_dir / bias_token(v)) for v in points]


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
    from ..identity import StageRef
    from ..jobset.materialize import latest_attempt, run_dir
    from ..parse import detect
    from ..runfiles import find_by_role
    from .stages import TRANSPORT_STAGES

    ladder = {r.name: r.token
              for r in StageRef.ladder([s.name for s in task.stages])}
    out: List[Dict] = []
    for name in TRANSPORT_STAGES:
        token = ladder.get(name)
        fact: Dict = {"stage": name, "token": token}
        if token is None:                      # the description omits it
            fact["state"] = "not_described"
            out.append(fact)
            continue
        container = base / token
        att = latest_attempt(container)
        if att is None:
            fact["state"] = "not_run"
            out.append(fact)
            continue
        fact["attempt"] = str(att.relative_to(base))
        outs = sorted(find_by_role(run_dir(container), ".out"),
                      key=lambda q: q.stat().st_mtime, reverse=True)
        if not outs:
            fact["state"] = "no_output"
            out.append(fact)
            continue
        try:
            res = detect(str(outs[0])).parse(str(outs[0]))
        except Exception as exc:               # a refusal is an ANSWER here
            fact["state"] = "unreadable"
            fact["why"] = str(exc)
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
            # 2026-09-18 for exactly this.
            if name in ("electrode_L", "electrode_R"):
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
    from ..jobset.materialize import latest_attempt, run_dir
    from .compose import PROVENANCE_FILE

    base = Path(base_dir)
    points_out: List[Dict] = []
    pending: List[Dict] = []
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
        current = None
        # `.out` IS the catalogue's role, so the catalogue finds it; the
        # newest-first order is this caller's own question and stays here.
        from ..runfiles import find_by_role
        for out in sorted(find_by_role(where, ".out"),
                          key=lambda p: p.stat().st_mtime, reverse=True):
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
            "current_a": current,
        })
    if not points_out:
        raise RecordError(
            "no transmission point has produced output yet -- run the "
            "transmission stage first:\n"
            "    molbuilder jobset prep run transmission && "
            "molbuilder jobset launch run transmission\n"
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
        "points": points_out,
        "iv": {
            "voltages_v": [p["bias_v"] for p in points_out],
            "current_a": [p["current_a"] for p in points_out],
        },
        "provenance": {
            "slot": provenance,
            "atom_permutation": "atom-permutation.json",
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
    """The printed deliverable: one row per point — G(E_F) and the
    engine's own current."""
    lines = [f"transport record — {record['label']}: "
             f"{len(record['points'])} point(s)"
             + (f", {len(record['pending'])} pending"
                if record.get("pending") else "")]
    lines.append(f"  {'V [V]':>8}  {'G(E_F) [G0]':>12}  {'I [A]':>12}")
    for p in record["points"]:
        g = p["conductance_g0"]
        i = p["current_a"]
        lines.append(
            f"  {p['bias_v']:>8.3f}  "
            + (f"{g:>12.4f}" if g is not None else f"{'--':>12}")
            + "  "
            + (f"{i:>12.4e}" if i is not None else f"{'--':>12}"))
    for p in record.get("pending", ()):
        lines.append(f"  {p['bias_v']:>8.3f}  {'pending':>12}  "
                     f"{'':>12}  ({p['why']})")
    return "\n".join(lines)
