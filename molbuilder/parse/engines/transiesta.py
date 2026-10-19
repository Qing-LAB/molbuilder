"""WHERE A SIESTA-FAMILY RUN IS, IN WORDS -- the one sentence the monitor's
status line and the run door's *running* detail say it in (`model/parse.md`
§ 2b, ``doing``; `running-a-job.md` § 4.2).

Two readings of the output speak: the row in flight's phase and iteration
(:func:`scf_words`), and -- until a TranSIESTA device's first SCF row, the
stretch it spends computing or reading the leads' surface Green's functions
-- its set-up (:func:`setup_words`), from the facts the reading pass keeps
in ``runtime_info["transiesta"]``.  Which speaks is the reader's one rule
(`siesta_reader.SiestaReader._where_words`), read by its `now()` for the
monitor and its `finish()` for the ending.  Travels beside the job with the
monitor (`runwrap.MONITOR_COMPANIONS`): the standard library alone.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


def scf_words(phase: Optional[str], cycle: Optional[int]) -> str:
    """``negf SCF iteration 12`` -- the phase and the iteration an SCF row
    states; ``""`` with neither."""
    return " ".join(b for b in (
        phase, "SCF" if (phase or cycle is not None) else "",
        f"iteration {cycle}" if cycle is not None else "") if b)


def setup_words(ts: Optional[Dict[str, Any]]) -> str:
    """TranSIESTA's set-up as it stands, in words -- the reading pass's
    ``runtime_info["transiesta"]`` (`model/parse.md` § 4a): which lead's
    surface Green's function it is computing and the file's estimated size
    (``Estimated file size``), each lead's function computed or -- a later
    frame's point at the same voltage -- reused (``Using GF-file``), the
    charge distribution at the switch; ``""`` when the output states none
    of it.  Read until the first SCF row; from then on the row speaks
    (:func:`scf_words`) -- a first point's periodic rows come between the
    leads' files and the charge line, a point starting from a ``.TSDE``
    goes to the charge line at once."""
    if not ts:
        return ""
    elecs: Dict[str, Dict[str, Any]] = ts.get("electrodes") or {}
    for name, el in elecs.items():
        if el.get("gf_started") and not el.get("gf_done"):
            size = el.get("gf_file_size")
            return (f"computing the {name} lead's surface Green's function"
                    + (f" (estimated {size})" if size else ""))
    done = [n for n, el in elecs.items() if el.get("gf_done")]
    reused = [n for n, el in elecs.items() if el.get("gf_reused")]
    if done or reused:
        if done and reused:
            words = (f"the {done[0]} lead's surface Green's function "
                     f"computed, the {reused[0]} lead's reused")
        else:
            names, verb = (done, "computed") if done else (reused, "reused")
            words = (f"the {names[0]} lead's surface Green's function {verb}"
                     if len(names) == 1
                     else f"the leads' surface Green's functions {verb}")
        if ts.get("charge_at_switch"):
            words += "; the charge distribution at the switch"
        return words
    if any("principal_cell" in el for el in elecs.values()):
        return "checking the leads' cells"
    return ""
