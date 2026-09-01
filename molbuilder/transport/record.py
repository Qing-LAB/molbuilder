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
    columns — E in eV (relative to E_F when the deck said
    ``TS.TBT.Erange.RelToEF T``, which the composite's deck does) and
    the k-averaged T(E).
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
        for out in sorted(where.glob("*.out"),
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
