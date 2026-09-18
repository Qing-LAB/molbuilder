"""What solver path a SIESTA run took -- the regexes, and a cheap way to ask.

Contract: `model/parse.md` § 2b.  This module owns the DIAG patterns and
nothing else; both readers share them, which is P-S4's "one reader per
question" made structural rather than aspirational.  The shape is
:mod:`~molbuilder.parse.engines._run_ending`'s, for the same reason: the
full parser builds Frames and needs numpy, a caller that wants the solver
line does not.

**SIESTA prints the question twice and they are different facts.**

* ``redata:`` echoes what was REQUESTED, after SIESTA's own parser has
  normalised or rejected it -- so it is what the run was told to do.
* ``diag:`` reports what the solver ACTUALLY used, which can differ:
  ELPA falls back, a GPU kernel is refused, a block size is clamped.

`bench/result.py` compares the two (``compare_asked_to_ran``), so both
have to be readable and they have to be told apart.  Until 2026-09-18 they
were read in two files under ONE key name -- the parser recorded
``siesta_diag['elpa_gpu']`` from ``redata:`` and `bench` recorded
``eff['elpa_gpu']`` from ``diag:`` -- so the same name meant *asked* in one
place and *ran* in the other.
"""
from __future__ import annotations

import re
from typing import Any, Dict

#: REQUESTED -- what SIESTA was told, echoed back after its own parsing.
REQUESTED = {
    "algorithm": re.compile(
        r"^\s*redata:\s+(?:Diagonalization\s+algorithm|Diag\.Algorithm)"
        r"\s*=\s*(\S+)", re.IGNORECASE | re.MULTILINE),
    "elpa_gpu": re.compile(
        r"^\s*redata:\s+(?:ELPA[.\s]?GPU|Diag\.ELPA\.GPU)\s*=\s*"
        r"([TF]|\.?true\.?|\.?false\.?)", re.IGNORECASE | re.MULTILINE),
}

#: RAN -- what the solver actually used.
RAN = {
    "algorithm": re.compile(r"^\s*diag:\s*Algorithm\s+=\s*(\S+)",
                            re.MULTILINE),
    "elpa_gpu_key": re.compile(
        r"^\s*diag:\s*ELPA GPU string key\s+=\s*(\S+)", re.MULTILINE),
    "processory_blocksize": re.compile(
        r"^\s*\*\s*ProcessorY,\s*Blocksize:\s*(\d+)\s+(\d+)", re.MULTILINE),
}

#: The ELPA-CUDA runtime banner, printed once on the first diagonalization.
GPU_DEVICE = re.compile(
    r"^\s*ELPA:\s+(?:NVIDIA\s+)?GPU\s+(?:detected|in use)\s*:\s*(.+?)"
    r"(?:\s+\((sm_\d+)\))?\s*$", re.IGNORECASE | re.MULTILINE)


def ran_facts(text: str) -> Dict[str, Any]:
    """What the solver actually used, from a SIESTA ``.out``'s text.

    Stdlib only, one pass per pattern.  Keys are absent when the run did
    not print them -- a missing fact is not a false one.
    """
    out: Dict[str, Any] = {}
    m = RAN["algorithm"].search(text)
    if m:
        out["diag_algorithm"] = m.group(1)
    m = RAN["elpa_gpu_key"].search(text)
    if m:
        out["elpa_gpu"] = m.group(1)
    m = RAN["processory_blocksize"].search(text)
    if m:
        # The line prints ProcessorY first, then Blocksize.
        out["blocksize"] = int(m.group(2))
    return out
