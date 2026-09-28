"""Run-directory ATOM-METADATA recovery — the results-side bridge.

Module: ``parse/dirs`` (directory-level composers — the ONE parse
layer allowed to touch the filesystem).  The text-level extraction is
``script_emit._extract_atom_metadata_dict``, which takes a STRING and
does no I/O — so the glob-and-read half lives here, and the block
grammar lives with the emitter that writes it (`plan.md` § 5d).

*This read ``AtomMetadataTextParser.parse(text).atom_metadata`` until
2026-09-05 — a class whose entire body built a ten-field ScriptResult so
this line could take one field back out.*

Callers:
  * ``web/blueprints/watch.py::_atom_metadata_json`` — the Results-tab
    load adapter (``/api/watch/load``).
  * ``tests/test_atom_metadata_results_bridge.py`` — the end-result
    seam tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from molbuilder.script_emit import _extract_atom_metadata_dict


def atom_metadata_json_for_run_dir(
    run_dir: Union[str, Path, None], n_atoms: Optional[int] = None
) -> Optional[str]:
    """Recover a run directory's embedded per-atom metadata block as a JSON
    string, ready to apply onto a loaded structure.

    The Build tab writes region labels / frozen tags / annotation channels
    into the ATOM-METADATA block of the input script it emits (``.fdf`` for
    SIESTA, ``.py`` for PySCF).  A results-side consumer that holds only the
    run's *output* geometry -- e.g. the Results-tab trajectory inspector,
    which loads coordinates from ``.molwatch.log`` / ``.out`` -- calls this
    to recover the metadata and re-apply it through
    ``script_emit.apply_atom_metadata`` -- THE one reader of this block,
    shared with the transport composite -- so the loaded view carries the same
    regions/frozen/annotations.

    This is a TRUSTED FRAGMENT, not a standalone ``.molstruct.json`` file:
    the block is molbuilder's OWN emit and, by design, omits the sidecar
    file envelope's integrity fields (``structure_hash``).  It must therefore
    be applied via ``apply_atom_metadata``, NOT validated through
    ``molstruct.load_text`` (which demands the full untrusted-file envelope).

    The block carries ONLY atom-scoped keys (schema_version / n_atoms_total /
    regions / frozen_atoms / selection_rules / annotations -- never cell /
    axis_kind / vacuum), so applying it never disturbs the consumer's own
    geometry or lattice.

    ``n_atoms``, when given, guards the block against the consumer's
    structure: a mismatch means the block's 0-based indices no longer point
    at the same atoms, so ``None`` is returned rather than metadata that
    would make ``apply_atom_metadata`` raise.

    Returns ``None`` when ``run_dir`` is falsy / not a directory, no input
    script carries a non-empty block, or the atom count disagrees.  Never
    raises -- a results view must still show coordinates when metadata
    recovery fails.
    """
    if not run_dir:
        return None
    try:
        d = Path(run_dir)
        if not d.is_dir():
            return None
        # SIESTA (.fdf) first, then PySCF (.py).  Staged SIESTA runs share
        # one metadata block across stages, so the first script carrying a
        # non-empty block wins; later scripts are consulted only if earlier
        # ones are absent or empty.
        # THE ROLES COME FROM THE CATALOGUE.  `.fdf` and `.py` are
        # `runfiles.WRITTEN`'s, one per engine, spelled here as globs until
        # 2026-09-08 (`project-layout.md` § 4.5).
        from ...runfiles import find_by_role
        scripts = find_by_role(d, ".fdf") + find_by_role(d, ".py")
    except OSError:                                         # pragma: no cover
        return None
    for script in scripts:
        try:
            text = script.read_text(encoding="utf-8-sig", errors="replace")
        except OSError:                                    # pragma: no cover
            continue
        md = _extract_atom_metadata_dict(text)
        if not md:
            continue
        if not (md.get("regions") or md.get("annotations")):
            continue                       # empty block -> try the next script
        if n_atoms is not None and md.get("n_atoms_total") != n_atoms:
            continue                       # indices no longer match -> skip
        try:
            return json.dumps(md)
        except (TypeError, ValueError):                    # pragma: no cover
            return None
    return None


def engine_offset_record_for_run_dir(
    run_dir: Union[str, Path, None]) -> Optional[dict]:
    """The ENGINE-OFFSET record of a run directory's own deck -- where its
    atoms were placed, the cell, the axis kinds -- or ``None``.

    The sibling of :func:`atom_metadata_json_for_run_dir`, over the same
    scripts found by the same roles: a record block in the deck the run was
    prepped from, read through ``script_emit.extract_engine_offset``, its one
    reader (`model/structure-periodicity.md` § 6.0).  The Results tab asks it
    instead of searching the directory for a `.source` pair a ladder keeps at
    its root -- the search that found nothing on 2026-09-25 and drew a box the
    engine never had.  ``None`` for a run made before the record existed; never
    raises.
    """
    if not run_dir:
        return None
    try:
        d = Path(run_dir)
        if not d.is_dir():
            return None
        from ...runfiles import find_by_role
        scripts = find_by_role(d, ".fdf") + find_by_role(d, ".py")
    except OSError:                                         # pragma: no cover
        return None
    from ...script_emit import extract_engine_offset
    for script in scripts:
        try:
            text = script.read_text(encoding="utf-8-sig", errors="replace")
        except OSError:                                    # pragma: no cover
            continue
        record = extract_engine_offset(text)
        if record:
            return record
    return None


def engine_frame_for_run_dir(
    run_dir: Union[str, Path, None],
    lattice: Optional[list] = None,
) -> Optional[Dict[str, Any]]:
    """The frame the engine had in a run directory -- ``cell``, ``axis_kind``,
    ``vacuum`` when known, and ``engine_offset`` -- or ``None`` when nothing
    there says (`model/structure-periodicity.md` § 6.0).

    THE ONE COMPOSER of a run's frame, for every door that makes a structure
    from an engine's output: the Results tab's trajectory door (the output's
    ``lattice`` passed in) and the codec, for an engine's own structure file
    that has no sidecar (`workingcopy_structure.StructureCodec.load`).  It
    was the trajectory door's private helper until 2026-09-27, when the
    second door -- SIESTA's ``<label>.xyz`` opened in the Results tab --
    turned out to state no frame at all (plan § 0a, M1's review).

    The CELL from the run's own output (``lattice``), or, when the output
    carries none (a PySCF log), the cell the deck placed the atoms in, from
    its record; the AXIS KINDS from the run's own deck -- its ENGINE-OFFSET
    record, the kinds as the structure had them when the deck was written --
    or, for a run made before the record, from the ``.source`` pair
    (job-contracts § 6.3); and the ORIGIN the engine's: these coordinates are
    its own, so the frame states an offset of 0 and the box sits at their
    origin.

    Until the trajectory door composed this on the server the browser
    composed ``{cell}`` alone, the axis kinds never reached the viewer, and
    an export from the Results tab stamped a lattice-bearing junction
    ``isolated`` on every axis -- the BDT-Au111 frame38 export that surfaced
    the hole.  A run made before the ``.source`` convention has no pair; the
    cell still travels, and the load door's own rule (a stated cell over
    never-stated axes derives periodic) covers the rest.

    Never raises -- a broken pair degrades to the lattice-only frame rather
    than taking a load down.
    """
    out: Dict[str, Any] = {}
    if isinstance(lattice, list) and len(lattice) == 3:
        out["cell"] = lattice
    # THE RUN'S OWN DECK SAYS WHAT THE ENGINE HAD: its ENGINE-OFFSET record
    # carries the axis kinds as the structure had them when the deck was
    # written (user, 2026-09-25: "we should show the axis_info as in
    # structure"), and the cell it placed the atoms in.
    record = engine_offset_record_for_run_dir(run_dir)
    if record:
        out["axis_kind"] = [str(k) for k in record.get("axis_kind") or []]
        if "cell" not in out and record.get("cell") is not None:
            out["cell"] = record["cell"]
    elif run_dir:
        # A RUN MADE BEFORE THE RECORD: the kinds from the `.source` pair, the
        # catalogue's role for *"the structure the calculation is of"* -- and
        # the frame-free facts only, the kinds and the vacuum.
        try:
            from ...runfiles import find_by_role
            pairs = find_by_role(Path(run_dir), ".source.xyz")
            if pairs:
                from ...workingcopy_structure import StructureCodec
                s = StructureCodec().read(pairs[0])
                if s.axis_kind:
                    out["axis_kind"] = list(s.axis_kind)
                if s.vacuum is not None:
                    out["vacuum"] = [float(v) for v in s.vacuum]
                if "cell" not in out and s.cell is not None:
                    out["cell"] = [[float(x) for x in row] for row in s.cell]
        except Exception:                        # noqa: BLE001
            pass
    # THESE COORDINATES ARE THE ENGINE'S, so they state its origin: 0, the
    # box at the origin of the frames on screen (§ 6.0) -- drawn verbatim, and
    # saved that way by an export, never re-centred.
    if "cell" in out:
        out["engine_offset"] = [0.0, 0.0, 0.0]
    return out or None


__all__ = ["atom_metadata_json_for_run_dir", "engine_frame_for_run_dir",
           "engine_offset_record_for_run_dir"]
