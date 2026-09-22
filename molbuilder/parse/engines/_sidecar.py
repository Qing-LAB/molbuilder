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

    # THE RUNG FALLBACK -- one sidecar in the directory is THIS run's.
    #
    # Everything above composes a NAME and tests it.  That answers
    # `bdt.out` and fails every laddered artifact: the sidecar is written
    # once per calculation and stemmed on the bare label, so nothing
    # composed from `bdt_01_coarse` can name `bdt.molstruct.json`, and
    # stripping the rung needs a label the caller usually does not pass.
    # Measured 2026-09-08 and again 2026-09-17: `bdt.out` returned
    # `{5, 6, 7}` and `bdt_01_coarse.out`, `.molwatch.log`, `.pyscf.log`
    # and `-run0.out` all returned nothing -- so "Hide frozen atoms" and
    # `runtime_info["frozen_atoms"]` were empty for EVERY staged run.
    #
    # Guessing the label is a bug and stays one (see above).  Looking is
    # not guessing: `project-layout.md` 1.4 -- a directory is a container
    # or a RUN, and a run holds "what that invocation produced, and
    # nothing else holds that" -- so a lone sidecar beside the artifact
    # is that calculation's, whatever it is called.
    #
    # This is `_siesta_fdf_path_for`'s own pattern, in this same module:
    # try the exact stem, then fall back to a single `*.fdf` in the
    # directory.  `model/parse.md` 5.3 names that function as the shape
    # a companion lookup may legitimately take -- it stays path-only and
    # does not invert 5's rule.  Asked through the framework's search
    # (`sidecars_in`), not a hand-rolled glob (`project-layout.md` 4.5).
    #
    # STRICTLY ADDITIVE: it runs only where the answer was already
    # nothing, and only when the directory is unambiguous.  Two sidecars
    # or none and this returns empty exactly as before.
    if sidecar_path is None:
        from molbuilder.sidecars.molstruct import SUFFIX, sidecars_in
        found = sidecars_in(base or ".")
        if len(found) == 1:
            # AND its label must be a PREFIX of this artifact's, on a `_`
            # boundary.  That is the half that costs nothing: it refuses an
            # unrelated lone sidecar (`other.molstruct.json` beside
            # `bdt_01_coarse.out`) while keeping every rung.  What it cannot
            # decide is `sample_02_test.out` beside `sample.molstruct.json`
            # -- measured, `runfiles.parse` reads a stage off BOTH spellings,
            # so no rule over the NAME separates them.  Only the directory
            # does, and 1.4 is what lets it.
            sc_stem = found[0].name[: -len(SUFFIX)]
            if sc_stem.endswith(".source"):
                sc_stem = sc_stem[: -len(".source")]
            if any(_s == sc_stem or _s.startswith(sc_stem + "_")
                   for _s in stems):
                sidecar_path = str(found[0])

    if sidecar_path is None:
        return set()
    # THROUGH THE DOOR, not around it.  `molstruct.load` is the sidecar
    # reader (`model/structure-molstruct.md`), and this function imported it
    # on the very next line to call `frozen_atoms` while reading the bytes
    # itself -- so the envelope was never validated and, with no `encoding=`
    # at all, a non-ASCII region label decoded under the platform locale.
    # `_load` reads `utf-8-sig` (BOM-tolerant) and validates.
    #
    # THE CONTRACT IS UNCHANGED: "empty set on any failure".  It raises
    # `MolstructJsonError`, a ValueError subclass, and wraps OSError in one,
    # so the same except clause still answers nothing for a sidecar that is
    # missing, malformed, or of a different structure.
    from molbuilder.sidecars import molstruct
    try:
        data = molstruct.load(sidecar_path)
    except (OSError, ValueError):
        return set()
    return set(molstruct.frozen_atoms(data))


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
        # A DIRECTORY NAMED `*.fdf` IS NOT A DECK.  Without the file test one
        # sitting beside the real deck makes this see two candidates and
        # answer None -- the frozen atoms lost with a single deck present.
        fdfs = [
            p for p in (os.path.join(base, f) for f in os.listdir(base))
            if p.lower().endswith(".fdf") and os.path.isfile(p)
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
            text = fh.read()
    except OSError:
        return set()

    # ONE DECK READER (`parse/fdf.py`).  Its `_norm` is fdf's real keyword
    # rule, so `Geometry_Constraints` and `Geometry-Constraints` are found as
    # well -- the two spellings the regex here was blind to.
    from ..fdf import _norm, _parse_fdf
    _scalars, blocks = _parse_fdf(text)
    rows = blocks.get(_norm("Geometry.Constraints"))
    if not rows:
        return set()

    frozen_one_based: Set[int] = set()
    for row in rows:
        line = " ".join(row)
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

    # SIESTA's `.fdf` writes constraints 1-based; translate back to the
    # 0-based Structure identity through the engine index API (never a bare
    # n - 1, which would be wrong for a 0-based engine).  The SAME sentence
    # and the SAME call as `read_frozen_atoms_from_siesta_out` above -- two
    # readers of one fact, and until 2026-09-22 only one of them routed.
    from ...engine_atom_index import from_engine_index
    return {from_engine_index(n, "siesta") for n in frozen_one_based}
