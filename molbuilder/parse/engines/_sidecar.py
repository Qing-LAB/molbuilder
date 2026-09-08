"""Helpers: read ``frozen_atoms`` indices from sources adjacent to a
trajectory file.

Absorbed from the legacy ``molbuilder.parsers._sidecar``, deleted with
that package on 2026-06-21 -- this is the only copy (provenance:
`docs/archive/old_docs/protocols/parse-module.md` § 8).

Three sources are supported.  All return a set of 0-based int indices;
the empty set means "no frozen atoms known from this source."  A
caller (typically the absorbed SIESTA parser body) consults them in
order and uses the first non-empty result:

  1. ``read_frozen_atoms_from_siesta_out(out_path)`` — read from the
     SIESTA ``.out``'s own ``siesta: Constraints applied in the
     following order:`` echo.  AUTHORITATIVE for SIESTA runs because
     it's what the engine actually applied.
  2. ``read_frozen_atoms(traj_path)`` — read from the
     ``.molstruct.json`` sidecar next to the trajectory.  Convention
     used by all engines.  Sidecar contract lives in
     ``docs/model/structure-molstruct.md``.
  3. ``read_frozen_atoms_from_siesta_fdf(traj_path)`` — read from a
     sibling SIESTA ``.fdf`` input file's ``%block Geometry
     .Constraints`` block.  Last-resort fallback when the .out lacks
     the constraints echo and there is no sidecar.

All functions return the empty set on any failure — missing file,
parse error, missing block.  Frozen-atom data is optional UI metadata;
a failure here must not break trajectory loading.
"""

from __future__ import annotations

import json as _json
import os
import re
from typing import Set

from ...runfiles import parse as _rf_parse


def read_frozen_atoms(traj_path: str, label: str = "") -> Set[int]:
    """Return frozen-atom 0-based indices from the sidecar next to
    ``traj_path``.  Looks for several naming conventions used across
    engines (``<stem>.molstruct.json``, with ``_optim`` and ``_geom``
    suffix-strip fallbacks for PySCF/geomeTRIC outputs)."""
    base, fname = os.path.split(traj_path)
    stem = fname
    for _role in ("_optim.xyz", ".xyz", ".pyscf.log", ".out",
                  ".molwatch.log"):
        if stem.endswith(_role):
            stem = stem[: -len(_role)]
            break
    stems = [stem]
    if stem.endswith("_geom"):
        stems.append(stem[:-len("_geom")])
    # A RUN ARTIFACT CARRIES THE RUNG AND THE ATTEMPT; THE SIDECAR DOES NOT.
    # The sidecar is written once for the calculation and stemmed on the bare
    # label (`job-contracts.md` § 2.2a: it carries no stage token, which is
    # what "carried" means), so a stem taken off `<label>_<stage>-run0.out`
    # never matched it.  MEASURED 2026-09-08: frozen atoms came back for
    # `bdt.out` and empty for `bdt_01_coarse.out`.
    #
    # THE LABEL IS REQUIRED TO STRIP THE RUNG, AND GUESSING IT IS A BUG.  A
    # first version stripped any `_<NN>_<name>` tail off the stem.  From a
    # filename alone that is not decidable: `bdt_01_coarse.out` (rung
    # `01_coarse` of `bdt`) and `sample_02_test.out` (an UNSTAGED calculation
    # whose label simply reads that way) are the same shape, and the guess
    # handed the second one a DIFFERENT calculation's sidecar -- measured
    # 2026-09-08, `{5, 6, 7}` where the answer is nothing.  That is the exact
    # ambiguity `runfiles.parse` refuses to resolve without a label, and the
    # reason it takes one.  So: with a label, strip exactly; without one, do
    # not strip at all.  `-run<N>` is peeled either way -- it is a declared
    # counter (`runfiles.QUALIFIERS`), unambiguous at the tail.
    for _s in list(stems):
        _bare = _RUN_INDEX_SUFFIX_RE.sub("", _s)
        if label and _bare.startswith(label + "_"):
            _parsed = _rf_parse(_bare + ".x", label)
            if _parsed is not None and _parsed.stage:
                _bare = label
        if _bare != _s:
            stems.append(_bare)
    candidates = [os.path.join(base, f"{_s}{sfx}")
                  for _s in stems
                  # `.source.molstruct.json` is what a PREPPED bundle holds
                  # (`web/handover-procedure.md`); `.molstruct.json` is what a
                  # structure folder holds.  Both are real, so both are tried.
                  for sfx in (".molstruct.json", ".source.molstruct.json")]
    sidecar_path = next(
        (p for p in candidates if os.path.isfile(p)), None)
    if sidecar_path is None:
        return set()
    try:
        with open(sidecar_path, "r", errors="replace") as fh:
            data = _json.load(fh)
    except (OSError, ValueError):
        return set()
    from molbuilder.sidecars import molstruct
    return set(molstruct.frozen_atoms(data))


_FDF_BLOCK_START_RE = re.compile(
    r"^\s*%block\s+Geometry\.?Constraints\b", re.IGNORECASE,
)
_FDF_BLOCK_END_RE = re.compile(
    r"^\s*%endblock\s+Geometry\.?Constraints\b", re.IGNORECASE,
)

_FDF_POSITION_KEYWORD_RE = re.compile(
    r"^\s*position\b\s*(.*)$", re.IGNORECASE,
)
_FDF_POSITION_RANGE_RE = re.compile(
    r"^\s*from\s+(\d+)\s+to\s+(\d+)"
    r"(?:\s+step\s+(\d+))?\s*$",
    re.IGNORECASE,
)


_RUN_INDEX_SUFFIX_RE = re.compile(r"-run\d+$")



_SIESTA_CONSTRAINTS_HEADER_RE = re.compile(
    r"siesta:\s+Constraints\s+applied\s+in\s+the\s+following\s+order:",
    re.IGNORECASE,
)
_SIESTA_CONSTRAINT_LINE_RE = re.compile(
    r"^\s*siesta:\s+Constraint\s*\(\d+\)\s*:\s*pos\s*$",
    re.IGNORECASE,
)
_SIESTA_CONSTRAINT_RANGES_RE = re.compile(
    r"^\s*\[\s*(.+?)\s*\]\s*$"
)
_RANGE_PIECE_RE = re.compile(r"(\d+)\s*--\s*(\d+)")


def read_frozen_atoms_from_siesta_out(out_path: str) -> Set[int]:
    """Return 0-based frozen-atom indices from the .out's own
    ``siesta: Constraints applied in the following order:`` echo.

    AUTHORITATIVE source of truth for SIESTA constraints — the data
    lives in the same file the Results-tab UI reads, so there's no
    filename-pairing heuristic between the .out and a sibling .fdf.

    Streams the file line-by-line and stops at the first non-
    constraints line after the section, so for the typical case
    (constraints near the top of the .out) we only touch the first
    few hundred KB regardless of total file size.
    """
    one_based: Set[int] = set()
    state = "before_header"
    expecting_data = False
    just_blanked = False
    try:
        fh = open(out_path, encoding="utf-8", errors="replace")
    except OSError:
        return set()
    try:
        for raw_line in fh:
            line = raw_line.rstrip("\n")

            if state == "before_header":
                if _SIESTA_CONSTRAINTS_HEADER_RE.search(line):
                    state = "in_section"
                continue

            if expecting_data:
                m_data = _SIESTA_CONSTRAINT_RANGES_RE.match(line)
                if m_data is None:
                    break
                body = m_data.group(1)
                for part in body.split(","):
                    part = part.strip()
                    if not part:
                        continue
                    m_range = _RANGE_PIECE_RE.match(part)
                    if m_range is not None:
                        start = int(m_range.group(1))
                        end = int(m_range.group(2))
                        if end >= start:
                            for n in range(start, end + 1):
                                one_based.add(n)
                    elif part.isdigit():
                        one_based.add(int(part))
                expecting_data = False
                just_blanked = False
                continue

            if _SIESTA_CONSTRAINT_LINE_RE.match(line):
                expecting_data = True
                just_blanked = False
                continue

            if not line.strip():
                if just_blanked:
                    break
                just_blanked = True
                continue

            break
    finally:
        fh.close()

    # SIESTA echoes constraints 1-based; translate back to the 0-based
    # Structure identity through the engine index API (never a bare n - 1,
    # which would be wrong for a 0-based engine).
    from ...engine_atom_index import from_engine_index
    return {from_engine_index(n, "siesta") for n in one_based}


def _siesta_fdf_path_for(traj_path: str) -> str | None:
    """Return the path of the SIESTA ``.fdf`` file most likely paired
    with ``traj_path``, or ``None`` if no candidate exists.

    Strips engine suffixes (``.out`` / ``.molwatch.log``) AND the
    wrapper's ``-run<N>`` index tail so ``foo-stage1-run3.out`` pairs
    with ``foo-stage1.fdf``.  Falls back to a single ``*.fdf`` in the
    same directory.
    """
    base, fname = os.path.split(traj_path)
    if not base:
        base = "."
    stem = fname
    for suffix in (".out", ".molwatch.log"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    stem = _RUN_INDEX_SUFFIX_RE.sub("", stem)
    same_stem = os.path.join(base, f"{stem}.fdf")
    if os.path.isfile(same_stem):
        return same_stem
    try:
        fdfs = [
            os.path.join(base, f) for f in os.listdir(base)
            if f.lower().endswith(".fdf")
        ]
    except OSError:
        return None
    if len(fdfs) == 1:
        return fdfs[0]
    return None


def read_frozen_atoms_from_siesta_fdf(traj_path: str) -> Set[int]:
    """Return frozen-atom 0-based indices parsed from the SIESTA
    ``.fdf`` paired with ``traj_path``'s ``Geometry.Constraints``
    block.

    SIESTA's ``.fdf`` uses 1-based atom indices; this function
    converts to 0-based for parity with the sidecar contract.
    Returns the empty set on any failure (no ``.fdf`` paired, block
    absent, parse error).
    """
    fdf_path = _siesta_fdf_path_for(traj_path)
    if fdf_path is None:
        return set()
    try:
        with open(fdf_path, "r", errors="replace") as fh:
            lines = fh.readlines()
    except OSError:
        return set()

    frozen_one_based: Set[int] = set()
    in_block = False
    for raw in lines:
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not in_block:
            if _FDF_BLOCK_START_RE.match(line):
                in_block = True
            continue
        if _FDF_BLOCK_END_RE.match(line):
            in_block = False
            continue
        m_kw = _FDF_POSITION_KEYWORD_RE.match(line)
        if m_kw is None:
            continue
        rest = m_kw.group(1).strip()
        if not rest:
            continue
        m_range = _FDF_POSITION_RANGE_RE.match(rest)
        if m_range is not None:
            start = int(m_range.group(1))
            stop = int(m_range.group(2))
            step = int(m_range.group(3) or 1)
            if step <= 0 or stop < start:
                continue
            for i in range(start, stop + 1, step):
                frozen_one_based.add(i)
            continue
        try:
            ints = [int(tok) for tok in rest.split()]
        except ValueError:
            continue
        for i in ints:
            if i >= 1:
                frozen_one_based.add(i)

    return {i - 1 for i in frozen_one_based}
