"""molbuilder.script_bundle -- workflow-handoff object.

Fuses a finished run's final structure with the labels and
annotations extracted from its originating .fdf / .py.  A bundle
materialized as ``<stem>.xyz`` + ``<stem>.molstruct.json`` lets the
next workflow stage pick the work up without dirging back to the
original design directory.

Contract: ``docs/protocols/bundle-contract.md``.

PR-A defines the dataclass + error type + API signatures.  The
final-coords readers and assembler implementations land in PR-B
(SIESTA `.XV`), PR-C (PySCF), and PR-D (materializer).  Today the
stubs raise :class:`NotImplementedError` with a pointer to the
follow-on task so callers can wire the typed seat now.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Python 3.8+ doesn't have typing.Literal until 3.8 but we target 3.10+.
from typing import Literal


class BundleError(Exception):
    """Raised by ``assemble_from_run_dir`` / ``write_bundle_as_handoff``
    on irrecoverable state.

    Bundle-contract.md § 8 enumerates the states that raise.  In
    short: missing-script, both-engines-present, atom-count
    mismatch, and (for the materializer) target-exists with
    ``overwrite=False``.
    """


@dataclass(frozen=True)
class RunBundle:
    """Portable workflow-handoff object.

    Field semantics per ``docs/protocols/bundle-contract.md § 3``.
    Frozen so callers can pass it around without defensive copies;
    construction-time validation lives in ``assemble_from_run_dir``.

    Notes on each field:

    * ``structure``: final coords + elements from the converged-run
      output (preferred) or initial coords (fallback).  Source
      recorded in ``final_coords_from``.
    * ``regions`` / ``frozen_atoms``: from the source script's
      ATOM-METADATA block.  Empty (``{}`` / ``[]``) when the script
      had no labels at generation time.
    * ``user_custom_lines``: inner lines of the USER-CUSTOM block
      verbatim (no ``#`` prefix stripping).  Empty list when the
      block was absent or empty.
    * ``provenance``: flat key/value extracted from PROVENANCE.
      Empty dict when the block was absent.
    * ``source_script``: absolute path to the .fdf / .py that fed
      extraction.
    * ``source_engine``: ``"siesta"`` for .fdf, ``"pyscf"`` for .py.
    * ``final_coords_from``: ``"xv"`` (SIESTA ``.XV``),
      ``"fdf-initial"`` (.fdf initial coords fallback when .XV
      missing), ``"py-opt"`` (PySCF ``<JOB>_optimized.xyz``),
      ``"py-initial"`` (.py mol.atom block fallback).
    * ``notes``: non-fatal diagnostics.  Always a (possibly empty)
      list; never ``None``.
    """
    # ``Structure`` typed via TYPE_CHECKING below to avoid a
    # heavyweight import at module-load time (molbuilder.structure
    # pulls in element tables + parsers).
    structure:         "Structure"  # type: ignore[name-defined]
    regions:           Dict[str, List[int]]
    frozen_atoms:      List[int]
    user_custom_lines: List[str]
    provenance:        Dict[str, str]
    source_script:     Path
    source_engine:     Literal["siesta", "pyscf"]
    final_coords_from: Literal["xv", "fdf-initial",
                               "py-opt", "py-initial"]
    notes:             List[str] = field(default_factory=list)


# --------------------------------------------------------------------- #
#  Assembler + materializer -- typed seats (PR-A) / impls (PR-B/C/D)    #
# --------------------------------------------------------------------- #


def assemble_from_run_dir(run_dir: Path) -> RunBundle:
    """Walk ``run_dir``, pick engine + final-coords source per the
    bundle-contract, fuse with ``ScriptSource`` from the originating
    script.  Raises :class:`BundleError` on irrecoverable state.

    See ``docs/protocols/bundle-contract.md § 4`` for source
    priority and § 8 for the error model.

    PR-B lands the SIESTA branch (``.fdf`` source script, ``.XV`` final
    coords, ``.fdf`` initial coords fallback).  PR-C will add the
    PySCF branch; the ambiguity check ("both engines in one dir")
    activates as soon as PR-C lands.
    """
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


def _assemble_siesta(run_dir: Path, fdf_paths: list[Path]) -> RunBundle:
    """SIESTA branch of assemble_from_run_dir.

    Picks the ``.fdf`` source per bundle-contract.md § 4.3 (largest
    by atom count, lexicographic tie-break), reads its ATOM-METADATA
    + USER-CUSTOM + PROVENANCE via script_contract.extract_script_source,
    and pairs it with the final-coords structure (``.XV`` preferred,
    fdf-initial fallback).  Validates atom-count consistency.
    """
    # Local imports keep the module's import-time graph thin.
    from molbuilder import script_contract as _sc
    from molbuilder.parsers.siesta_struct import (
        read_xv, read_fdf_initial_coords,
        SiestaXVError, SiestaFdfStructureError,
    )

    notes: list[str] = []

    # Pick the source .fdf.  Single .fdf is the common case; multi-
    # .fdf happens in staged runs (-stage1.fdf, -stage2.fdf, ...).
    # The contract picks "largest by atom count" so the rule is
    # invariant under stage re-ordering.  Read each .fdf's atom-md
    # n_atoms_total for the comparison; if a .fdf has no atom-md,
    # treat its atom count as the count we read from its initial
    # coords block (fallback).
    if len(fdf_paths) > 1:
        chosen, chosen_n = None, -1
        for cand in fdf_paths:
            try:
                txt = cand.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            md = _sc.extract_atom_metadata_dict(txt)
            if md is not None and isinstance(md.get("n_atoms_total"), int):
                n = int(md["n_atoms_total"])
            else:
                # Fallback: count atoms in the initial-coords block.
                try:
                    n = len(read_fdf_initial_coords(txt).elements)
                except SiestaFdfStructureError:
                    n = -1
            if n > chosen_n or (n == chosen_n and cand.name < (chosen.name if chosen else "")):
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
    source = _sc.extract_script_source(fdf_text)
    notes.extend(source.notes)
    if source.schema_version is not None and source.schema_version < 3:
        raise BundleError(
            f"{fdf_path.name}: atom-metadata schema_version "
            f"{source.schema_version} is older than v3; re-render the "
            f"source script with current molbuilder.")

    # Final-coords source: same stem .XV preferred, fdf-initial last.
    xv_path = fdf_path.with_suffix(".XV")
    structure = None
    final_coords_from: Literal["xv", "fdf-initial"] = "fdf-initial"
    if xv_path.exists():
        try:
            structure = read_xv(xv_path)
            final_coords_from = "xv"
        except SiestaXVError as exc:
            notes.append(
                f"{xv_path.name}: {exc}; falling back to .fdf initial "
                f"coords.")
    if structure is None:
        try:
            structure = read_fdf_initial_coords(fdf_text)
        except SiestaFdfStructureError as exc:
            raise BundleError(
                f"could not read final coords from {xv_path.name} OR "
                f"initial coords from {fdf_path.name}: {exc}") from exc
        if xv_path.exists():
            pass  # fallback note already in notes
        else:
            notes.append(
                f"no {xv_path.name} in run dir; bundle reflects "
                f"initial-coords from {fdf_path.name} -- NOT converged "
                f"geometry.  Re-run if you need optimized coords.")

    # Atom-count consistency check.  Only meaningful when the source
    # script DID carry an atom-metadata block (the unlabeled case is
    # silent on n_atoms_total -- bundle assembles fine).
    atom_md = _sc.extract_atom_metadata_dict(fdf_text)
    if atom_md is not None and isinstance(atom_md.get("n_atoms_total"), int):
        n_md = int(atom_md["n_atoms_total"])
        if n_md != len(structure.elements):
            raise BundleError(
                f"atom-metadata n_atoms_total ({n_md}) does not match "
                f"final structure atom count ({len(structure.elements)}). "
                f"The script and the final coords likely come from "
                f"different runs; re-render or clean the run dir.")

    return RunBundle(
        structure=structure,
        regions=source.regions or {},
        frozen_atoms=source.frozen_atoms or [],
        user_custom_lines=source.user_custom_lines or [],
        provenance=source.provenance or {},
        source_script=fdf_path.resolve(),
        source_engine="siesta",
        final_coords_from=final_coords_from,
        notes=notes,
    )


def _assemble_pyscf(run_dir: Path, py_paths: list[Path]) -> RunBundle:
    """PySCF branch of assemble_from_run_dir.

    Mirrors :func:`_assemble_siesta` for the PySCF engine.  Final
    coords come from ``<JOB>_optimized.xyz`` (preferred) or the
    ``.py`` ``mol = gto.M(atom = '''...''')`` block (fallback).
    """
    from molbuilder import script_contract as _sc
    from molbuilder.parsers.pyscf_struct import (
        read_optimized_xyz, read_py_initial_coords, extract_pyscf_job,
        PyscfStructureError,
    )

    notes: list[str] = []

    if len(py_paths) > 1:
        chosen, chosen_n = None, -1
        for cand in py_paths:
            try:
                txt = cand.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            md = _sc.extract_atom_metadata_dict(txt)
            if md is not None and isinstance(md.get("n_atoms_total"), int):
                n = int(md["n_atoms_total"])
            else:
                try:
                    n = len(read_py_initial_coords(txt).elements)
                except PyscfStructureError:
                    n = -1
            if n > chosen_n or (n == chosen_n and cand.name < (chosen.name if chosen else "")):
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
    source = _sc.extract_script_source(py_text)
    notes.extend(source.notes)
    if source.schema_version is not None and source.schema_version < 3:
        raise BundleError(
            f"{py_path.name}: atom-metadata schema_version "
            f"{source.schema_version} is older than v3; re-render the "
            f"source script with current molbuilder.")

    # Final-coords source: <JOB>_optimized.xyz preferred, py-initial last.
    structure = None
    final_coords_from: Literal["py-opt", "py-initial"] = "py-initial"
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
            structure = read_optimized_xyz(opt_path)
            final_coords_from = "py-opt"
        except (ValueError, OSError) as exc:
            notes.append(
                f"{opt_path.name}: {exc}; falling back to .py initial "
                f"coords.")
    if structure is None:
        try:
            structure = read_py_initial_coords(py_text)
        except PyscfStructureError as exc:
            raise BundleError(
                f"could not read final coords from {opt_path.name if opt_path else '<JOB>_optimized.xyz'} "
                f"OR initial coords from {py_path.name}: {exc}") from exc
        if opt_path is None:
            notes.append(
                f"no <JOB>_optimized.xyz in run dir; bundle reflects "
                f"initial-coords from {py_path.name} -- NOT converged "
                f"geometry.  Re-run if you need optimized coords.")

    atom_md = _sc.extract_atom_metadata_dict(py_text)
    if atom_md is not None and isinstance(atom_md.get("n_atoms_total"), int):
        n_md = int(atom_md["n_atoms_total"])
        if n_md != len(structure.elements):
            raise BundleError(
                f"atom-metadata n_atoms_total ({n_md}) does not match "
                f"final structure atom count ({len(structure.elements)}). "
                f"The script and the final coords likely come from "
                f"different runs; re-render or clean the run dir.")

    return RunBundle(
        structure=structure,
        regions=source.regions or {},
        frozen_atoms=source.frozen_atoms or [],
        user_custom_lines=source.user_custom_lines or [],
        provenance=source.provenance or {},
        source_script=py_path.resolve(),
        source_engine="pyscf",
        final_coords_from=final_coords_from,
        notes=notes,
    )


def write_bundle_as_handoff(bundle: RunBundle,
                            target_dir: Path,
                            *,
                            stem: str,
                            overwrite: bool = False
                            ) -> Tuple[Path, Path]:
    """Materialize ``bundle`` as ``<target_dir>/<stem>.xyz`` +
    ``<target_dir>/<stem>.molstruct.json``.

    Atomic via :func:`molstruct_json.save`.  ``overwrite=False``
    raises :class:`BundleError` when either destination file exists.

    Returns ``(xyz_path, sidecar_path)``.

    Not yet implemented -- lands in PR-D (task #491) alongside the
    L3 round-trip test.
    """
    raise NotImplementedError(
        "write_bundle_as_handoff lands in task #491 (PR-D)."
    )


__all__ = [
    "BundleError",
    "RunBundle",
    "assemble_from_run_dir",
    "write_bundle_as_handoff",
]
