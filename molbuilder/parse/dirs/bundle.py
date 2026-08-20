"""Run-bundle DirParser — directory → handoff-ready BundleResult.

H1 of parse-module.md migration (was Phase G wrapper around
``molbuilder.script_bundle.assemble_from_run_dir``): absorbed the
read-side assembler bodies directly so this module no longer imports
from ``molbuilder.script_bundle``.  The legacy module stays in place
until H4 — ``write_bundle_as_handoff`` (the materializer half of the
round-trip) still consumes ``RunBundle`` until H2 rehomes the write
side.

Dispatch design — NOT auto-registered with the DirParser registry.
Two reasons:

  1. ``JobDirParser`` already auto-claims any directory with a
     ``.fdf`` for JobMonitor decoding.  Bundle assembly is a
     DIFFERENT user intent — "extract the handoff bundle from this
     run dir" — and shouldn't fight for the dispatch slot.
  2. Bundle assembly raises :exc:`BundleError` on irrecoverable state
     (e.g. both ``.fdf`` and ``.py`` in the same dir).  That belongs
     at the explicit-call boundary, not in a heuristic ``can_parse``.

Callers use the class directly::

    from molbuilder.parse.dirs.bundle import BundleDirParser
    result = BundleDirParser.parse(run_dir)
    # result: BundleResult (structure + regions + frozen_atoms + ...)
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from molbuilder.parse.base import DirParser
from molbuilder.parse.coords.pyscf_geom import _read_optimized_xyz
from molbuilder.parse.coords.siesta_xv import (
    SiestaXVError,
    _read_xv,
    _read_xv_cell,
)
from molbuilder.parse.scripts.atom_metadata import (
    _extract_atom_metadata_dict,
)
from molbuilder.parse.scripts.provenance import _extract_provenance_dict
from molbuilder.parse.scripts.user_custom import _extract_user_custom_inner
from molbuilder.parse.types import BundleResult

from ._assembler_helpers import (
    PyscfStructureError,
    SiestaFdfStructureError,
    check_fdf_handedness,
    check_xv_handedness,
    extract_pyscf_job,
    extract_system_label,
    read_fdf_initial_coords,
    read_py_initial_coords,
)


class BundleError(Exception):
    """Raised by :meth:`BundleDirParser.parse` on irrecoverable state.

    bundle-contract.md § 8 enumerates the states that raise: missing
    engine script, both engines present in one dir, atom-count
    mismatch between source script and final coords.
    """


def _iso_z() -> str:
    return datetime.fromtimestamp(time.time(), tz=timezone.utc).isoformat(
        timespec="milliseconds").replace("+00:00", "Z")


# --------------------------------------------------------------------- #
#  Script-source extraction (regions + frozen + user-custom + ...)       #
# --------------------------------------------------------------------- #


#: The atom-metadata schema this build WRITES (script_emit stamps it from the
#: same constant). Compared against rather than a literal: a version written
#: down in two places is how the block came to claim v4 while carrying v7.
from molbuilder.sidecars.molstruct import READABLE_VERSIONS as _READABLE
from molbuilder.sidecars.molstruct import SCHEMA_VERSION as _CURRENT_SCHEMA


def _extract_script_source(text: str) -> Dict[str, Any]:
    """Single-pass extract over a generated-script body for the bundle
    layer.  Returns a dict with:

      * ``regions``           dict[str, list[int]] | None
      * ``frozen_atoms``      list[int] | None
      * ``user_custom_lines`` list[str] | None
      * ``provenance``        dict[str, str] | None
      * ``schema_version``    int | None
      * ``notes``             list[str]

    ``None`` distinguishes "block absent" from "block present but
    empty" (``{}`` / ``[]``) per bundle-contract.md § 5.1.  A block whose
    version is not the one this build writes is READ (a finished run must stay
    readable) and surfaced as a diagnostic note naming what may be missing --
    see the comment at the check itself.
    """
    notes: List[str] = []
    atom_md = _extract_atom_metadata_dict(text)
    regions: Optional[Dict[str, List[int]]] = None
    frozen: Optional[List[int]] = None
    schema_version: Optional[int] = None
    if atom_md is not None:
        sv = atom_md.get("schema_version")
        if isinstance(sv, int):
            schema_version = sv
            if sv not in _READABLE:
                # REFUSED, NOT READ (2026-08-01, by decision; amended
                # 2026-08-20 to the READABLE SET -- v8 added only optional
                # identity columns, so a v7 block reads whole and refusing
                # it would have made every existing finished run's labels
                # unreadable for a change that loses nothing).
                #
                # It used to READ an older block and attach a warning, on the
                # reasoning that a finished run cannot be re-exported the way a
                # sidecar can.  That reasoning is wrong for a product still
                # being built: an older block stores the same facts in different
                # places, so "read it and warn" hands back a payload that LOOKS
                # complete and quietly is not -- which is how a junction's fifty
                # frozen atoms came back as an empty list.  Supporting both
                # shapes also doubles what every reader, test and debugging
                # session has to hold in its head, for data that will be
                # regenerated anyway.
                #
                # The scripts get regenerated.  That is cheaper than a format
                # nobody can reason about.
                notes.append(
                    f"atom-metadata schema_version {sv}, but this molbuilder "
                    f"writes v{_CURRENT_SCHEMA} and reads "
                    f"{sorted(_READABLE)} only. The block was "
                    f"NOT read -- an older one keeps the same facts in "
                    f"different places (before v7 the frozen atoms sat in a "
                    f"top-level key rather than in `regions`), so reading it "
                    f"would silently drop what it cannot map. Re-generate the "
                    f"script from the structure."
                )
            else:
                # A BLOCK AT ANY OTHER VERSION IS READ, AND SAID SO ABOUT.
                #
                # This is a FINISHED RUN on disk, so refusing it outright would
                # make a user's existing results unreadable -- unlike the
                # sidecar, which is refused because it is still being worked on
                # and can be re-exported. But the note has to be accurate, and
                # this one was not: it said "molbuilder expects 4 — loading with
                # current handler", which reads as a formality.
                #
                # It is not. Before the label store was unified, the reserved
                # `frozen_atoms` list sat in a top-level key; this reader takes
                # the whole store from `regions`, so on an older block the
                # LABELS COME BACK AND THE FROZEN SET DOES NOT. That is how a
                # junction's fifty pinned electrode atoms read back as an empty
                # list. The note says which fact is at risk now, instead of
                # reporting a number.
                raw_regions = atom_md.get("regions")
                if isinstance(raw_regions, dict):
                    regions = {
                        str(k): sorted({int(i) for i in v})
                        for k, v in raw_regions.items()
                    }
                else:
                    regions = {}
                # ONE designated read: v5 keeps the reserved label in
                # `regions`, v3/v4 kept it in a top-level key, and this
                # knows which without the caller spelling the name.
                from molbuilder.sidecars import molstruct as _ms
                frozen = _ms.frozen_atoms(atom_md)
        else:
            notes.append(
                "atom-metadata block has no schema_version; ignored.")
    return {
        "regions":           regions,
        "frozen_atoms":      frozen,
        "user_custom_lines": _extract_user_custom_inner(text),
        "provenance":        _extract_provenance_dict(text),
        "schema_version":    schema_version,
        "notes":             notes,
    }


# --------------------------------------------------------------------- #
#  SIESTA branch assembler                                               #
# --------------------------------------------------------------------- #


def _assemble_siesta(run_dir: Path,
                     fdf_paths: List[Path]) -> Dict[str, Any]:
    """SIESTA branch.  Picks the ``.fdf`` source per bundle-contract.md
    § 4.3 (largest by atom count, lexicographic tie-break), reads its
    atom-metadata + user-custom + provenance, pairs with final-coords
    (``.XV`` preferred, fdf-initial fallback).  Validates atom-count
    consistency.  Returns the bundle fields the caller stitches into a
    :class:`BundleResult`."""
    notes: List[str] = []

    # Pick the source .fdf.  Multi-.fdf happens in staged runs
    # (-stage1.fdf, -stage2.fdf, ...).  Contract: largest by atom
    # count so the rule is invariant under stage re-ordering.
    if len(fdf_paths) > 1:
        chosen, chosen_n = None, -1
        for cand in fdf_paths:
            try:
                txt = cand.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            md = _extract_atom_metadata_dict(txt)
            if md is not None and isinstance(md.get("n_atoms_total"), int):
                n = int(md["n_atoms_total"])
            else:
                # Fallback: count atoms in the initial-coords block.
                try:
                    n = len(read_fdf_initial_coords(txt).elements)
                except SiestaFdfStructureError:
                    n = -1
            if n > chosen_n or (
                    n == chosen_n
                    and cand.name < (chosen.name if chosen else "")):
                chosen, chosen_n = cand, n
        if chosen is None:
            raise BundleError(
                f"none of the .fdf files in {run_dir} were readable.")
        fdf_path = chosen
        notes.append(
            f"multiple .fdf in run_dir; picked {fdf_path.name} "
            f"(largest by atom count: {chosen_n}).")
    else:
        fdf_path = fdf_paths[0]

    fdf_text = fdf_path.read_text(encoding="utf-8", errors="replace")
    source = _extract_script_source(fdf_text)
    notes.extend(source["notes"])
    sv = source["schema_version"]
    if sv is not None and sv < 3:
        raise BundleError(
            f"{fdf_path.name}: atom-metadata schema_version {sv} is "
            f"older than v3; re-render the source script with current "
            f"molbuilder.")

    # Final-coords source: <SystemLabel>.XV preferred, fdf-initial last.
    # SIESTA names output files by the in-body SystemLabel directive,
    # NOT by the .fdf basename.  Try in priority: SystemLabel-derived
    # -> basename-derived -> any single *.XV glob (handles renames).
    system_label = extract_system_label(fdf_text)
    xv_candidates: List[Path] = []
    if system_label is not None:
        xv_candidates.append(run_dir / f"{system_label}.XV")
    xv_candidates.append(fdf_path.with_suffix(".XV"))
    xv_path: Optional[Path] = None
    for cand in xv_candidates:
        if cand.exists():
            xv_path = cand
            break
    if xv_path is None:
        glob_hits = sorted(run_dir.glob("*.XV"))
        if len(glob_hits) == 1:
            xv_path = glob_hits[0]
            notes.append(
                f"no SystemLabel-derived .XV found; picked "
                f"{xv_path.name} (the only *.XV in the run dir).")
        elif len(glob_hits) > 1:
            notes.append(
                f"multiple *.XV in run_dir ({[p.name for p in glob_hits]}) "
                f"and SystemLabel ({system_label!r}) doesn't disambiguate "
                f"-- falling back to .fdf initial coords.")
    structure = None
    final_coords_from = "fdf-initial"
    if xv_path is not None:
        try:
            structure = _read_xv(xv_path)
            final_coords_from = "xv"
        except SiestaXVError as exc:
            # XV file exists but is unreadable.  Combine the
            # parse-error reason with the loud "this is not converged
            # geometry" call-out in a single note.
            notes.append(
                f"{xv_path.name}: {exc}; falling back to "
                f".fdf initial coords -- NOT converged geometry.  "
                f"Re-run if you need optimized coords.")
    if structure is None:
        try:
            structure = read_fdf_initial_coords(fdf_text)
        except SiestaFdfStructureError as exc:
            xv_name = xv_path.name if xv_path else "<SystemLabel>.XV"
            raise BundleError(
                f"could not read final coords from {xv_name} OR "
                f"initial coords from {fdf_path.name}: {exc}") from exc
        if xv_path is None:
            notes.append(
                f"no .XV in run dir; bundle reflects initial-coords "
                f"from {fdf_path.name} -- NOT converged geometry.  "
                f"Re-run if you need optimized coords.")

    # Atom-count consistency check.  Only meaningful when the source
    # script DID carry an atom-metadata block.
    atom_md = _extract_atom_metadata_dict(fdf_text)
    if atom_md is not None and isinstance(atom_md.get("n_atoms_total"), int):
        n_md = int(atom_md["n_atoms_total"])
        if n_md != len(structure.elements):
            raise BundleError(
                f"atom-metadata n_atoms_total ({n_md}) does not match "
                f"final structure atom count ({len(structure.elements)}). "
                f"The script and the final coords likely come from "
                f"different runs; re-render or clean the run dir.")

    # Handedness diagnostic.  A left-handed cell silently mirrors the
    # structure when Fractional coords project through it; for chiral
    # molecules this is wrong physics.  Soft (notes-only) because the
    # cell COULD be intentional; loud warning text gives the user the
    # obligation to look.
    if final_coords_from == "xv" and xv_path is not None:
        warn = check_xv_handedness(xv_path)
        if warn:
            notes.append(warn)
    fdf_handedness_warn = check_fdf_handedness(fdf_text)
    if fdf_handedness_warn:
        notes.append(fdf_handedness_warn)

    # Cell — try the .XV's leading 3 rows when we used XV; otherwise
    # leave None.  (The legacy assembler returned a RunBundle that
    # didn't carry cell at all; absorbing here lets BundleResult
    # surface it from the same .XV that gave us positions.)
    cell: Optional[np.ndarray] = None
    if final_coords_from == "xv" and xv_path is not None:
        cell = _read_xv_cell(xv_path)

    return {
        "structure":    structure,
        "cell":         cell,
        "regions":      source["regions"] or {},
        "frozen_atoms": source["frozen_atoms"] or [],
        "notes":        notes,
        "schema_version": source["schema_version"],
        "source_engine":     "siesta",
        "source_script":     fdf_path.resolve(),
        "final_coords_from": final_coords_from,
    }


# --------------------------------------------------------------------- #
#  PySCF branch assembler                                                #
# --------------------------------------------------------------------- #


def _assemble_pyscf(run_dir: Path,
                    py_paths: List[Path]) -> Dict[str, Any]:
    """PySCF branch.  Mirrors :func:`_assemble_siesta`.  Final coords
    come from ``<JOB>_optimized.xyz`` (preferred) or the ``.py``
    ``mol = gto.M(atom = '''...''')`` block (fallback)."""
    notes: List[str] = []

    if len(py_paths) > 1:
        chosen, chosen_n = None, -1
        for cand in py_paths:
            try:
                txt = cand.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            md = _extract_atom_metadata_dict(txt)
            if md is not None and isinstance(md.get("n_atoms_total"), int):
                n = int(md["n_atoms_total"])
            else:
                try:
                    n = len(read_py_initial_coords(txt).elements)
                except PyscfStructureError:
                    n = -1
            if n > chosen_n or (
                    n == chosen_n
                    and cand.name < (chosen.name if chosen else "")):
                chosen, chosen_n = cand, n
        if chosen is None:
            raise BundleError(
                f"none of the .py files in {run_dir} were readable.")
        py_path = chosen
        notes.append(
            f"multiple .py in run_dir; picked {py_path.name} "
            f"(largest by atom count: {chosen_n}).")
    else:
        py_path = py_paths[0]

    py_text = py_path.read_text(encoding="utf-8", errors="replace")
    source = _extract_script_source(py_text)
    notes.extend(source["notes"])
    sv = source["schema_version"]
    if sv is not None and sv < 3:
        raise BundleError(
            f"{py_path.name}: atom-metadata schema_version {sv} is "
            f"older than v3; re-render the source script with current "
            f"molbuilder.")

    # Final-coords source: <JOB>_optimized.xyz preferred, py-initial last.
    structure = None
    final_coords_from = "py-initial"
    job = extract_pyscf_job(py_text)
    opt_path: Optional[Path] = None
    if job:
        cand = run_dir / f"{job}_optimized.xyz"
        if cand.exists():
            opt_path = cand
    if opt_path is None:
        # Fallback: glob for any *_optimized.xyz in the dir.  The
        # JOB-derived path is preferred (deterministic); the glob
        # rescues cases where the script was edited after generation
        # and the JOB literal drifted from the file naming.
        candidates = sorted(run_dir.glob("*_optimized.xyz"))
        if len(candidates) == 1:
            opt_path = candidates[0]
        elif len(candidates) > 1:
            notes.append(
                f"multiple *_optimized.xyz in run_dir "
                f"({[c.name for c in candidates]}); JOB literal in "
                f"{py_path.name} doesn't disambiguate -- falling back "
                f"to py-initial coords.")
    if opt_path is not None:
        try:
            structure = _read_optimized_xyz(opt_path)
            final_coords_from = "py-opt"
        except (ValueError, OSError) as exc:
            # Combine the parse-error reason with the loud "NOT
            # converged geometry" warning so the user sees both.
            notes.append(
                f"{opt_path.name}: {exc}; falling back to .py initial "
                f"coords -- NOT converged geometry.  Re-run if you "
                f"need optimized coords.")
    if structure is None:
        try:
            structure = read_py_initial_coords(py_text)
        except PyscfStructureError as exc:
            opt_name = opt_path.name if opt_path else "<JOB>_optimized.xyz"
            raise BundleError(
                f"could not read final coords from {opt_name} "
                f"OR initial coords from {py_path.name}: {exc}") from exc
        if opt_path is None:
            notes.append(
                f"no <JOB>_optimized.xyz in run dir; bundle reflects "
                f"initial-coords from {py_path.name} -- NOT converged "
                f"geometry.  Re-run if you need optimized coords.")

    atom_md = _extract_atom_metadata_dict(py_text)
    if atom_md is not None and isinstance(atom_md.get("n_atoms_total"), int):
        n_md = int(atom_md["n_atoms_total"])
        if n_md != len(structure.elements):
            raise BundleError(
                f"atom-metadata n_atoms_total ({n_md}) does not match "
                f"final structure atom count ({len(structure.elements)}). "
                f"The script and the final coords likely come from "
                f"different runs; re-render or clean the run dir.")

    return {
        "structure":    structure,
        "cell":         None,            # PySCF outputs carry no cell
        "regions":      source["regions"] or {},
        "frozen_atoms": source["frozen_atoms"] or [],
        "notes":        notes,
        "schema_version": source["schema_version"],
        "source_engine":     "pyscf",
        "source_script":     py_path.resolve(),
        "final_coords_from": final_coords_from,
    }


def _assemble_from_run_dir(run_dir: Path) -> Dict[str, Any]:
    """Walk ``run_dir``, pick engine, dispatch to the matching
    branch.  Returns the bundle fields the caller stitches into a
    :class:`BundleResult`.  Raises :exc:`BundleError` on irrecoverable
    state (no engine script, both engines present in one dir)."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise BundleError(f"run_dir does not exist: {run_dir}")
    fdf_paths = sorted(run_dir.glob("*.fdf"))
    py_paths  = sorted(run_dir.glob("*.py"))
    if fdf_paths and py_paths:
        raise BundleError(
            f"{run_dir.name} contains both .fdf and .py; bundle source "
            f"is ambiguous.  Clean up or split the directory before "
            f"bundling.")
    if not fdf_paths and not py_paths:
        raise BundleError(
            f"no engine script (.fdf or .py) in {run_dir}")
    if py_paths:
        return _assemble_pyscf(run_dir, py_paths)
    return _assemble_siesta(run_dir, fdf_paths)


# --------------------------------------------------------------------- #
#  DirParser wrapper                                                     #
# --------------------------------------------------------------------- #


class BundleDirParser(DirParser):
    """Assemble a run directory into a portable handoff bundle.

    Returns :class:`BundleResult` carrying the final-coords structure
    + the script-contract ATOM-METADATA labels (regions +
    frozen_atoms) + the user-custom block + the diagnostic notes.

    NOT auto-registered (see module docstring); call explicitly.
    """

    name   = "bundle-dir"
    label  = ("molbuilder run-bundle handoff "
              "(absorbed assemble_from_run_dir)")
    output = BundleResult

    @classmethod
    def can_parse(cls, run_dir: Path) -> bool:
        """A directory bundleable iff it has at least one ``.fdf`` or
        ``.py`` engine script.  Mirrors the assembler's precondition
        exactly; the ambiguity check (both engines present →
        BundleError) only fires in :meth:`parse`, not here.
        """
        try:
            for child in run_dir.iterdir():
                if child.is_file() and (
                    child.name.endswith(".fdf")
                    or child.name.endswith(".py")
                ):
                    return True
        except OSError:
            return False
        return False

    @classmethod
    def parse(cls, run_dir: Path) -> BundleResult:
        """Run the absorbed assembler + wrap in a typed
        :class:`BundleResult`.  Propagates :exc:`BundleError` upward
        unchanged."""
        bundle = _assemble_from_run_dir(Path(run_dir))
        try:
            source_str = str(Path(run_dir).resolve())
        except OSError:
            source_str = str(run_dir)
        block_schema_versions: Dict[str, int] = {}
        sv = bundle.get("schema_version")
        if isinstance(sv, int):
            block_schema_versions["atom-metadata"] = sv
        source_script = bundle.get("source_script")
        source_script_str = (
            str(source_script) if source_script is not None else None
        )
        return BundleResult(
            schema_version=1,
            parsed_at=_iso_z(),
            parser_name=cls.name,
            source=source_str,
            structure=bundle["structure"],
            cell=bundle["cell"],
            regions=dict(bundle["regions"] or {}),
            frozen_atoms=list(bundle["frozen_atoms"] or []),
            notes=list(bundle["notes"] or []),
            block_schema_versions=block_schema_versions,
            source_engine=bundle.get("source_engine"),
            final_coords_from=bundle.get("final_coords_from"),
            source_script=source_script_str,
        )


__all__ = ["BundleDirParser", "BundleError"]
