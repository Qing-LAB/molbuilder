"""Shared helper: read `frozen_atoms` from a `.molstruct.json` sidecar.

The same conventions used by `parsers/pyscf.py::_read_sidecar_frozen_atoms`
(2026-06-11) — but promoted out of the PySCF parser so SIESTA + molwatch
parsers can reuse it without copy-paste.

The sidecar contract lives in `docs/protocols/sidecar-contract.md`.
Each parser passes the trajectory file path; this helper figures out
the conventional sidecar location and returns the frozen-atom index
list as a sorted set of 0-based ints.

Empty return when:
  * no sidecar is present
  * sidecar parses but has no ``frozen_atoms`` field
  * sidecar fails to parse (caller decides whether that's fatal —
    typically not, since the field is optional)
"""

from __future__ import annotations

import json as _json
import os
from typing import Set


def read_frozen_atoms(traj_path: str) -> Set[int]:
    """Return frozen-atom 0-based indices from the sidecar next to
    ``traj_path``.  Looks for several naming conventions used across
    engines (``<stem>.molstruct.json``, with ``_optim`` and ``_geom``
    suffix-strip fallbacks for PySCF/geomeTRIC outputs)."""
    base, fname = os.path.split(traj_path)
    stem = fname
    # PySCF / geomeTRIC convention: foo_optim.xyz pairs with foo.molstruct.json.
    if stem.endswith("_optim.xyz"):
        stem = stem[: -len("_optim.xyz")]
    elif stem.endswith(".xyz"):
        stem = stem[: -len(".xyz")]
    elif stem.endswith(".out"):
        stem = stem[: -len(".out")]
    elif stem.endswith(".molwatch.log"):
        stem = stem[: -len(".molwatch.log")]
    # Try a couple of conventions: the geomeTRIC pair often has a
    # stripped ``_geom`` suffix on the sidecar too.
    candidates = [
        os.path.join(base, f"{stem}.molstruct.json"),
    ]
    if stem.endswith("_geom"):
        candidates.append(
            os.path.join(base, stem[:-5] + ".molstruct.json"))
    sidecar_path = next(
        (p for p in candidates if os.path.isfile(p)), None)
    if sidecar_path is None:
        return set()
    try:
        with open(sidecar_path, "r", errors="replace") as fh:
            data = _json.load(fh)
    except (OSError, ValueError):
        return set()
    frozen = data.get("frozen_atoms")
    if not isinstance(frozen, list):
        return set()
    return {int(i) for i in frozen if isinstance(i, int)}
