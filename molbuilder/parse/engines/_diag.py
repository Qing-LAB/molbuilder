"""What solver path a SIESTA run took -- the regexes, and a cheap way to ask.

Contract: `model/parse.md` § 2b.  This module owns the DIAG patterns, so a
pattern has one home instead of two.

**It is one home, NOT a shared table.**  The two halves have one importer
each -- `REQUESTED`/`GPU_DEVICE` by `engines/siesta.py`, `ran_facts` (over
`RAN`) by `bench/result.py` -- and neither reads the other's half.  So
nothing here makes the two READERS agree; it makes neither able to drift
from a pattern written elsewhere.  *(This claimed "both readers share them,
which is P-S4's one reader per question made structural" until 2026-09-18.
No object in this module has two readers.)*  The shape is
:mod:`~molbuilder.parse.engines._run_ending`'s, for the same reason: the
full parser builds Frames and needs numpy, a caller that wants the solver
line does not.

**SIESTA prints the question twice and they are different facts.**

* ``redata:`` echoes what was REQUESTED, after SIESTA's own parser has
  normalised or rejected it -- so it is what the run was told to do.
* ``diag:`` reports what the solver ACTUALLY used, which can differ:
  ELPA falls back, a GPU kernel is refused, a block size is clamped.

They were read in two files with two private pattern sets until 2026-09-18,
under one key name: the parser recorded ``siesta_diag['elpa_gpu']`` from
``redata:`` and `bench` recorded ``eff['elpa_gpu']`` from ``diag:``, so the
same name meant *asked* in one place and *ran* in the other.  The PATTERNS
are single-homed here now and both readers import them.  The two output keys
still share the name in their own dicts; they are never merged, so it is a
naming hazard rather than a collision.
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
