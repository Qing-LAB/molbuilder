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
        read_xv, read_fdf_initial_coords, extract_system_label,
        check_xv_handedness, check_fdf_handedness,
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

    # Final-coords source: <SystemLabel>.XV preferred, fdf-initial last.
    # SIESTA names output files by the in-body SystemLabel directive,
    # NOT by the .fdf basename.  molbuilder's own generator emits
    # stage-suffixed .fdf names (``h2-stage2.fdf``) over a single
    # SystemLabel (``h2``), so ``.with_suffix(".XV")`` would miss the
    # real ``h2.XV``.  Try in priority: SystemLabel-derived ->
    # basename-derived -> any single *.XV glob (handles renamed files).
    system_label = extract_system_label(fdf_text)
    xv_candidates: list[Path] = []
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
    final_coords_from: Literal["xv", "fdf-initial"] = "fdf-initial"
    if xv_path is not None:
        try:
            structure = read_xv(xv_path)
            final_coords_from = "xv"
        except SiestaXVError as exc:
            # XV file exists but is unreadable.  Combine the two
            # warnings so the user sees BOTH the parse-error reason
            # AND the loud "this is not converged geometry" call-out
            # in a single note (pre-fix the second half was suppressed
            # because the xv-exists branch took `pass`).
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

    # Handedness diagnostic.  A left-handed cell silently mirrors
    # the structure when Fractional coords project through it; for
    # chiral molecules this is wrong physics.  We check both the
    # source of the final coords AND the .fdf's LatticeVectors
    # block.  Soft (notes-only, doesn't refuse to bundle) because
    # the cell COULD be intentional; loud warning text gives the
    # user the obligation to look.
    if final_coords_from == "xv" and xv_path is not None:
        warn = check_xv_handedness(xv_path)
        if warn:
            notes.append(warn)
    fdf_handedness_warn = check_fdf_handedness(fdf_text)
    if fdf_handedness_warn:
        notes.append(fdf_handedness_warn)

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
            # Same shape as the SIESTA fix above: combine the
            # parse-error reason with the loud "NOT converged
            # geometry" warning so the user sees both.
            notes.append(
                f"{opt_path.name}: {exc}; falling back to .py initial "
                f"coords -- NOT converged geometry.  Re-run if you "
                f"need optimized coords.")
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

    The next tab's existing ``.xyz`` load path (build / spectra /
    transport) picks the pair up unchanged: it reads the XYZ for
    structure, then ``apply_sidecar_if_possible`` picks the sidecar
    up for labels.  No new load primitive needed downstream.

    ``overwrite=False`` raises :class:`BundleError` when either
    destination file already exists.  Writes are atomic
    (tmp + rename) for both files individually; if the .xyz lands
    but the sidecar write fails partway through, the caller is
    left with a labeled-but-no-sidecar XYZ — recoverable but a
    diagnostic note is added downstream.

    Returns ``(xyz_path, sidecar_path)``.

    Contract: ``docs/protocols/bundle-contract.md § 5.2 + § 8``.
    """
    from molbuilder.parsers import molstruct_json as _msj

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    xyz_path     = target_dir / f"{stem}.xyz"
    sidecar_path = _msj.sidecar_path_for(xyz_path)
    if not overwrite:
        for p in (xyz_path, sidecar_path):
            if p.exists():
                raise BundleError(
                    f"target exists: {p}.  Pass overwrite=True to "
                    f"replace, or pick a different stem/target_dir.")

    # Provenance line on the .xyz so a curious user reading the
    # XYZ comment can trace the bundle back to its source.
    comment = (
        f"bundled from {bundle.source_script.name} "
        f"({bundle.source_engine}/{bundle.final_coords_from})"
    )
    xyz_text = bundle.structure.to_xyz(comment=comment)
    _atomic_write_text(xyz_path, xyz_text)

    # structure_hash is the SHA-256 of the XYZ file BYTES (per
    # molstruct_json.sha256_of_file); compute from the freshly-
    # written file so the invariant pin is what apply_sidecar_if_possible
    # will check on load.
    structure_hash = _msj.sha256_of_file(xyz_path)
    sidecar_payload = _msj.to_dict(
        n_atoms_total=len(bundle.structure.elements),
        structure_hash=structure_hash,
        regions=bundle.regions or None,
        frozen_atoms=bundle.frozen_atoms or None,
        created_by="molbuilder script_bundle",
    )
    _msj.save(sidecar_path, sidecar_payload)
    return (xyz_path, sidecar_path)


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically (tmp + rename) so a
    partial write never leaves a half-XYZ on disk.

    Tmp filename is unique per writer (PID + monotonic-ns) so two
    concurrent writers to the same final path don't collide on the
    same tmp file (a deterministic ``<path>.tmp`` would let writer
    B's open(...) truncate writer A's in-flight bytes before either
    finishes, and the os.replace would commit a mix).
    """
    import os, time
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(
        path.suffix + f".{os.getpid()}.{time.monotonic_ns()}.tmp"
    )
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


__all__ = [
    "BundleError",
    "RunBundle",
    "assemble_from_run_dir",
    "write_bundle_as_handoff",
]
